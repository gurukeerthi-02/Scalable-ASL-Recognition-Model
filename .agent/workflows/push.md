---
description: Steps to push project changes to the remote repository
---

Follow these steps to push your local changes to the GitHub repository:

1. **Check the status of your changes**
   Verify which files have been modified or are new.
   ```bash
   git status
   ```

2. **Stage your changes**
   Add all changes to the staging area.
   ```bash
   git add .
   ```

3. **Commit your changes**
   Create a commit with a descriptive message.
   ```bash
   git commit -m "Update project: [Your description here]"
   ```

4. **Pull latest changes (Optional but Recommended)**
   Ensure your local branch is up to date with the remote to avoid conflicts.
   ```bash
   git pull origin main
   ```

5. **Push to the remote repository**
   Push your commits to the `main` branch on GitHub.
   // turbo
   ```bash
   git push origin main
   ```

> **Note:** If you are working on a different branch, replace `main` with your branch name. You can check your current branch using `git branch`.
