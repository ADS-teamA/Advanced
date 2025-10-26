# Branch Protection Setup Guide

This repository requires branch protection on `main` to ensure code quality and proper review process.

## 🔒 Required Settings

Since CODEOWNERS file is now in place, you need to enable branch protection rules.

### Quick Setup (GitHub Web UI)

1. Go to: **Settings** → **Branches** → **Add branch protection rule**
2. Branch name pattern: `main`
3. Configure the following settings:

#### Required Settings:
- ✅ **Require a pull request before merging**
  - Required number of approvals before merging: **1**
  - ✅ Dismiss stale pull request approvals when new commits are pushed
  - ✅ Require approval from Code Owners (ensures @michalprusek must approve)

- ✅ **Require status checks to pass before merging** (optional, if you have CI/CD)
  - You can add specific status checks later

- ✅ **Do not allow bypassing the above settings**
  - This ensures even admins must follow the rules

#### Recommended Additional Settings:
- ✅ **Require conversation resolution before merging** (ensures all PR comments are addressed)
- ✅ **Require linear history** (prevents merge commits, keeps history clean)

#### NOT Recommended:
- ❌ **Allow force pushes** - Should stay OFF
- ❌ **Allow deletions** - Should stay OFF

### What This Achieves:

✅ **No direct commits to main** - All changes must go through pull requests
✅ **Required approval from @michalprusek** - CODEOWNERS ensures you're automatically assigned as reviewer
✅ **Stale approvals dismissed** - New commits require re-approval
✅ **Protected from force pushes** - History cannot be rewritten

### Alternative: GitHub CLI Setup

If you have admin permissions, you can set this up via CLI:

```bash
gh api \
  --method PUT \
  repos/ADS-teamA/Advanced/branches/main/protection \
  --input - <<'EOF'
{
  "required_status_checks": null,
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
EOF
```

### Verification

After setting up, verify with:

```bash
gh api repos/ADS-teamA/Advanced/branches/main/protection
```

You should see the protection rules listed.

---

**Status:** CODEOWNERS file is configured ✅
**Action Required:** Enable branch protection via GitHub Settings ⚠️
