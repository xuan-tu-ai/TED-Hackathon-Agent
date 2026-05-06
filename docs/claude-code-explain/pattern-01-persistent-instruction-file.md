# Pattern 1 — Persistent Instruction File

> **Tóm tắt:** Commit một file text vào repo; harness tự động load nó vào mọi agent session như là "luật lệ" cố định — không cần user nhắc lại mỗi lần.

---

## Vấn đề nó giải quyết

Mỗi lần bắt đầu một session mới, model không nhớ gì về project của bạn. Nếu không có cơ chế "luật cố định", bạn phải lặp lại context trong từng session:

- "Đây là Rust project, dùng `cargo clippy` để check"
- "Đừng tự overwrite `CLAUDE.md`"
- "Tests nằm ở `tests/`, cập nhật song song với `src/`"

Đây là ma sát không cần thiết, và nó âm thầm gây lỗi — ngày bạn quên nhắc là ngày agent làm sai.

**Giải pháp:** commit luật của bạn cùng với code. Nếu nó nằm trong repo, nó luôn ở đó.

---

## Source code trong repo

| File | Vai trò |
|------|---------|
| [`CLAUDE.md`](../CLAUDE.md) | Instruction file ở root — luật áp dụng toàn repo |
| [`rust/CLAUDE.md`](../rust/CLAUDE.md) | Instruction file ở sub-workspace — luật riêng cho Rust |
| [`rust/crates/runtime/src/prompt.rs:43–44`](../rust/crates/runtime/src/prompt.rs#L43) | Hằng số giới hạn bộ nhớ: `MAX_INSTRUCTION_FILE_CHARS = 4_000`, `MAX_TOTAL_INSTRUCTION_CHARS = 12_000` |
| [`rust/crates/runtime/src/prompt.rs:48–51`](../rust/crates/runtime/src/prompt.rs#L48) | Struct `ContextFile` — đại diện cho 1 file đã load (path + content) |
| [`rust/crates/runtime/src/prompt.rs:203–224`](../rust/crates/runtime/src/prompt.rs#L203) | `discover_instruction_files()` — leo cây thư mục, tìm tất cả candidate |
| [`rust/crates/runtime/src/prompt.rs:330–351`](../rust/crates/runtime/src/prompt.rs#L330) | `render_instruction_files()` — render các file thành section trong prompt, có budget enforcement |
| [`rust/crates/runtime/src/prompt.rs:144–166`](../rust/crates/runtime/src/prompt.rs#L144) | `SystemPromptBuilder::build()` — điều phối toàn bộ thứ tự assembly của system prompt |

---

## Mindset

Hãy nghĩ `CLAUDE.md` như `.editorconfig` hay `Makefile` — một config file của project tự động định hình hành vi của tooling mà không cần setup thủ công mỗi lần. Điểm khác biệt là `CLAUDE.md` định hình hành vi của *agent*.

Ba đặc tính tạo nên sức mạnh của pattern này:

**1. Được versioned.** Khi luật thay đổi, git history ghi lại lý do. Rollback chỉ là một lệnh `git revert`.

**2. Được scoped.** File ở repo root áp dụng toàn bộ. File ở `rust/CLAUDE.md` chỉ áp dụng khi working directory nằm trong `rust/`. Monorepo tự động có luật riêng cho từng package — không cần cấu hình thêm.

**3. Được compose.** Nhiều file cùng được load và ghép lại. Luật của root + luật của sub-workspace + `.claw/instructions.md` cùng tồn tại mà không xung đột.

---

## Technical walkthrough

### Bước 1 — Session start kích hoạt discovery

Khi harness bắt đầu một session, nó gọi `ProjectContext::discover_with_git(cwd)`. Hàm này ngay lập tức khởi động quá trình tìm instruction files, bắt đầu từ working directory hiện tại.

```
cwd = /home/user/projects/myrepo/rust/crates/runtime
```

### Bước 2 — Leo cây ancestor (từ root xuống)

`discover_instruction_files()` xây dựng danh sách đường đi từ filesystem root đến `cwd`:

```
["/", "/home", "/home/user", "/home/user/projects",
 "/home/user/projects/myrepo",
 "/home/user/projects/myrepo/rust",
 "/home/user/projects/myrepo/rust/crates",
 "/home/user/projects/myrepo/rust/crates/runtime"]
```

Danh sách được đảo ngược để bắt đầu từ root, đảm bảo luật cấp cao được load trước (scope gần nhất được append sau cùng → có độ ưu tiên cao hơn).

### Bước 3 — Kiểm tra 4 candidate mỗi thư mục

Với mỗi thư mục trong chuỗi, harness kiểm tra 4 path theo thứ tự:

```
{dir}/CLAUDE.md              ← file chính, commit vào repo
{dir}/CLAUDE.local.md        ← override cá nhân, thường gitignore
{dir}/.claw/CLAUDE.md        ← file chính trong claw config dir
{dir}/.claw/instructions.md  ← file override trong claw config dir
```

`CLAUDE.local.md` và `.claw/instructions.md` dùng cho override theo máy — thường được gitignore, không commit lên team. `CLAUDE.md` là phiên bản dùng chung của team.

Bất kỳ file nào tồn tại và có nội dung đều được thêm vào danh sách dưới dạng `ContextFile { path, content }`.

### Bước 4 — Deduplicate theo content hash

Trước khi trả về, các file được deduplicate bằng cách hash nội dung đã chuẩn hoá (gộp blank lines, strip whitespace). Nếu hai file ở hai thư mục khác nhau có nội dung giống hệt nhau, chỉ giữ lại file đầu tiên.

Điều này quan trọng trong monorepo khi `CLAUDE.md` ở root có thể được symlink hoặc copy vào sub-package.

### Bước 5 — Inject vào system prompt sau dynamic boundary

`SystemPromptBuilder::build()` lắp ráp system prompt theo thứ tự cố định sau:

```
[1] Intro section                          ← "You are an interactive agent..."
[2] Output style (optional)
[3] System rules                           ← tool behavior, permissions
[4] Doing tasks                            ← "đọc code trước khi sửa", "không thêm abstraction thừa"
[5] Actions with care                      ← reversibility / blast radius
─────────── __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__ ───────────
[6] Environment                            ← cwd, date, OS, model name
[7] Project context                        ← git status, recent commits, diff
[8] Claude instructions          ◄──── CLAUDE.md files được inject ở đây
[9] Runtime config                         ← nội dung settings.json
```

Marker `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` tách phần *static scaffolding* (sections 1–5, không bao giờ thay đổi) khỏi phần *dynamic context* (sections 6–9, được rebuild mỗi session). Điều này cho phép **prompt caching** ở tầng API: phần static được cache; chỉ phần dynamic được gửi mới.

### Bước 6 — Budget enforcement khi render

`render_instruction_files()` áp dụng hai giới hạn:

- **Mỗi file:** tối đa `MAX_INSTRUCTION_FILE_CHARS = 4_000` ký tự. File vượt quá sẽ bị truncate và thêm marker `[truncated]`.
- **Tổng tất cả file:** tối đa `MAX_TOTAL_INSTRUCTION_CHARS = 12_000` ký tự. Khi hết budget, các file còn lại bị bỏ qua với một note thông báo.

Mỗi file được render thành một `##` section riêng, kèm tên file và scope (thư mục cha), để model biết rule đến từ đâu:

```markdown
## CLAUDE.md (scope: /home/user/projects/myrepo)

## Detected stack
- Languages: Rust.
...

## CLAUDE.md (scope: /home/user/projects/myrepo/rust)

...
```

---

## Model thực sự nhận được gì

Đến thời điểm API call được thực hiện, system prompt của model có một section như sau:

```
# Claude instructions

## CLAUDE.md (scope: /home/user/projects/myrepo)

This file provides guidance to Claude Code...

## Detected stack
- Languages: Rust.

## Working agreement
- Prefer small, reviewable changes...
- Do not overwrite existing `CLAUDE.md` content automatically...

## CLAUDE.md (scope: /home/user/projects/myrepo/rust)

This file provides guidance to Claw Code...

## Working agreement
- Keep shared defaults in `.claw.json`...
```

Cả hai file được load, scoped đúng, trong budget.

---

## Nếu bạn tự build (bằng bất kỳ ngôn ngữ nào)

Pattern này không phụ thuộc vào vendor. Implementation cốt lõi chỉ là:

```python
# pseudo-code

def discover_instruction_files(cwd: Path) -> list[str]:
    # Leo từ root xuống cwd
    ancestors = list(reversed(cwd.parents)) + [cwd]
    candidates = []
    for dir in ancestors:
        for name in ["CLAUDE.md", "CLAUDE.local.md", ".claw/CLAUDE.md", ".claw/instructions.md"]:
            path = dir / name
            if path.exists() and path.read_text().strip():
                candidates.append({"path": path, "content": path.read_text()})
    return deduplicate_by_content_hash(candidates)

def build_system_prompt(cwd: Path) -> str:
    static_part = join([intro, system_rules, task_rules, actions])
    dynamic_part = join([environment, project_context,
                         *render_instruction_files(discover_instruction_files(cwd)),
                         config])
    return static_part + "\n\n__DYNAMIC_BOUNDARY__\n\n" + dynamic_part
```

Các quyết định thiết kế cần giữ nguyên:

| Quyết định | Lý do |
|-----------|-------|
| Load theo thứ tự root-first | Scope gần nhất được append sau → dễ override mà không xung đột |
| Giới hạn budget | Token budget có giới hạn; instruction file có thể phình to vô tội vạ |
| Inject sau dynamic boundary | Phần static không thay đổi → cacheable; chỉ phần dynamic mới tốn tiền |
| `CLAUDE.local.md` riêng biệt | Cho phép override cá nhân mà không ảnh hưởng team |

---

## Các cạm bẫy thường gặp

| Cạm bẫy | Hậu quả | Cách tránh |
|---------|---------|-----------|
| `CLAUDE.md` quá dài | Bị truncate ở 4,000 ký tự; model mất các luật ở cuối file | Giữ dưới 3,000 ký tự; tách nội dung chi tiết sang file khác |
| Luật mâu thuẫn giữa các scope | Model bị confused khi root nói "dùng tabs", sub-workspace nói "dùng spaces" | File scope gần hơn phải explicit override, không chỉ bổ sung thêm |
| Commit `CLAUDE.local.md` | Override cá nhân của bạn áp dụng cho cả team | Thêm `CLAUDE.local.md` vào `.gitignore` |
| Để secret trong `CLAUDE.md` | Gửi đến model API ở dạng plaintext mỗi session | Không bao giờ. Dùng env vars hoặc secrets manager |
| Viết luật bằng câu bị động | "Tests should be updated" → model có thể hiểu là optional | Dùng mệnh lệnh cách: "Update tests in `tests/` whenever you change `src/`" |
