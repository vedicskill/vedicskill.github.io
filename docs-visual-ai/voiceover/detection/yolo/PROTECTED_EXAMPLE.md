---
title: Protected Example Document
description: This is an example of a protected documentation page
requiresLogin: true
---

# Protected Example Document

This page is protected and will only display after a user logs in.

## Features

- This page requires authentication to view
- The protection is automatic based on the `requiresLogin: true` frontmatter
- No additional React components needed in the MDX
- Users will see a professional login prompt if they're not authenticated

## How It Works

1. When you access this page without logging in, you'll see a login prompt
2. Click the "Go to Login" button to navigate to the login page
3. Use demo credentials:
   - Username: `admin` or `user`
   - Password: any value
4. After logging in, you'll be automatically redirected back to this page

## Protected Content

This section is only visible to authenticated users. It demonstrates that the entire page content is protected.

```typescript
// This code is only visible to authenticated users
const secretData = {
  apiKey: "sk-***",
  internalNotes: "This is internal documentation"
};
```

## Logout

To log out, use the logout button in the top-right corner of the navbar. You'll be logged out and returned to the home page.
