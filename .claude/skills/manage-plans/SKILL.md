---
name: manage-plans
description: Use it when there's a task, a new feature, or a bug to fix, provided the workload is large. It will manage a sprint plan to break down the large task into atomic tasks, and continuously update progress until completion.
---

# Manage Plans Skill

## Purpose
This skill is designed to manage and execute sprint plans for large tasks, new features, or bug fixes. It breaks down complex work into atomic tasks, tracks progress, and ensures that all work is completed efficiently and effectively.

## Workflow

### 1. Create or Update Global Plan file

- **File Location:** `docs/PLAN.md`
- **Purpose:** This file serves as the single source of truth for the overall project plan. It should be updated whenever there are a new sprint plan or changes to existing plans.
- **Content:** The file should contain a high-level overview of the project, including current sprint goals, major milestones, and links to detailed sprint plans, not the detailed breakdown of tasks.


*Notes*: You can view an example of template in [_GLOBAL_PLAN_TEMPLATE.md](./references/_GLOBAL_PLAN_TEMPLATE.md).

### 2. Create or Update Sprint Plan file
- **File Location:** `docs/sprint-plans/sprint-<N>.md`
- **Purpose:** This file contains the detailed breakdown of tasks for a specific sprint. It should be created at the start of a sprint and updated as work progresses.
- **Content:** The file should include the sprint goal, an ordered list of atomic tasks, and the status of each task (pending, in progress, done). Status per task: `[ ]` pending / `[x]` done / `[~]` in progress.
- **What is an Atomic Task?** An atomic task is a task that focuses on a single concern and can be completed in one focused implementation step. If a task touches more than one concern, it should be split into multiple atomic tasks.

*Notes*: You can view an example of template in [_SPRINT_PLAN_TEMPLATE.md](./references/_SPRINT_PLAN_TEMPLATE.md).