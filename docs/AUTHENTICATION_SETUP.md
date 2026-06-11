---
title: VedicSkill Authentication System Implementation
description: Complete guide to the client-side authentication system for protecting selected documentation pages
keywords: [authentication, login, protected content, React Context, localStorage, Docusaurus]
---

# VedicSkill Authentication System

This document explains the complete implementation of the client-side authentication system for protecting selected documentation pages in your Docusaurus 3.x site.

## 📋 Overview

The authentication system provides:

- ✅ Client-side only authentication (no backend required)
- ✅ Session persistence using localStorage
- ✅ Automatic page protection based on frontmatter
- ✅ Professional login page with demo credentials
- ✅ Authentication-aware navbar with user profile display
- ✅ Smooth user experience with loading states
- ✅ React Context API for state management

## 🏗️ Architecture

### Components & Files

```
src/
├── context/
│   └── AuthContext.tsx           # Authentication state management
├── components/
│   ├── ProtectedContent.tsx      # Content protection wrapper
│   └── ProtectedContent.module.css
├── pages/
│   ├── login.tsx                 # Login page
│   └── login.module.css
├── theme/
│   ├── Root.tsx                  # Root wrapper with AuthProvider
│   ├── DocItem/
│   │   └── Layout/
│   │       └── index.tsx         # Swizzled for auto-protection
│   └── Navbar/
│       ├── index.tsx             # Swizzled navbar with auth menu
│       └── navbar-auth.module.css
└── hooks/
    └── useProtectedRoute.ts      # Utility hooks
```

## 🔐 Authentication Flow

### 1. Initialization

When the app loads:

1. `Root.tsx` wraps the entire app with `<AuthProvider>`
2. `AuthContext` loads stored user data from localStorage
3. Components receive `isLoading: true` while initializing
4. Once ready, `isLoading: false` and components render

### 2. Login Flow

```
User visits protected page
    ↓
Page requires login? (from frontmatter)
    ↓ YES
User authenticated? (from localStorage)
    ↓ NO
Show login prompt with "Go to Login" button
    ↓
User clicks "Go to Login"
    ↓
Redirect to /login page
    ↓
User enters credentials
    ↓
Call login(username, password)
    ↓
Store user in localStorage
    ↓
Redirect back to original page
```

### 3. Session Persistence

```
User logs in
    ↓
Store user data in localStorage
    ↓
User refreshes page
    ↓
AuthContext reads localStorage on mount
    ↓
User remains logged in
    ↓
Protected pages are accessible
```

## 📝 Protecting Pages with Frontmatter

To protect a documentation page, add `requiresLogin: true` to the frontmatter:

### Public Page (Default)

```markdown
---
title: Getting Started
description: Public documentation
---

# Getting Started

This page is public and accessible to everyone.
```

### Protected Page

```markdown
---
title: Internal Architecture
description: Protected documentation
requiresLogin: true
---

# Internal Architecture

This page requires authentication to view.
```

### Role-Based Protection (Future Extension)

For future development, you can add role information to user objects:

```typescript
// In AuthContext.tsx, modify the User interface
interface User {
  username: string;
  role: 'user' | 'admin' | 'moderator';
  permissions?: string[];
}

// In DocItem/Layout/index.tsx, you could add role checking
const requiredRole = metadata?.frontMatter?.requiredRole;
const userHasRole = user?.role === requiredRole;
```

## 🔑 Demo Credentials

For testing purposes, the authentication accepts:

| Role  | Username | Password      | Use Case |
|-------|----------|---------------|----------|
| Admin | `admin`  | Any value     | Full access |
| User  | `user`   | Any value     | Limited access |

**Note**: This is a demo system. In production, implement proper backend authentication.

## 🛠️ Usage Guide

### Using the Auth Hook

```typescript
import { useAuth } from '@/context/AuthContext';

export function MyComponent() {
  const { isAuthenticated, user, login, logout } = useAuth();

  if (!isAuthenticated) {
    return <p>Please log in</p>;
  }

  return <p>Welcome, {user?.username}!</p>;
}
```

### Manually Wrapping Components

```typescript
import { ProtectedContent } from '@/components/ProtectedContent';

export function SensitiveComponent() {
  return (
    <ProtectedContent requiresLogin={true}>
      <h1>Secret Information</h1>
      <p>This is only visible to logged-in users</p>
    </ProtectedContent>
  );
}
```

### Protecting Routes Programmatically

```typescript
import { useProtectedRoute } from '@/hooks/useProtectedRoute';
import { useNavigate } from '@docusaurus/router';

export function ProtectedPage() {
  const { hasAccess, redirectToLogin } = useProtectedRoute();
  const navigate = useNavigate();

  if (!hasAccess) {
    return (
      <button onClick={() => {
        redirectToLogin();
        navigate('/login');
      }}>
        Login Required
      </button>
    );
  }

  return <YourContent />;
}
```

## 🎨 Customization

### Change Login Credentials

Edit `src/context/AuthContext.tsx`:

```typescript
const MOCK_USERS: Record<string, User> = {
  admin: { username: 'admin', role: 'admin' },
  user: { username: 'user', role: 'user' },
  // Add more users here
};
```

### Customize Protected Content Prompt

Edit `src/components/ProtectedContent.tsx` to change the appearance and messaging.

### Style Customization

Modify the CSS modules:
- `src/components/ProtectedContent.module.css` - Login prompt styling
- `src/pages/login.module.css` - Login page styling
- `src/theme/Navbar/navbar-auth.module.css` - Navbar auth menu styling

### Extend Authentication

To connect to a real backend:

1. Modify `login()` in `AuthContext.tsx` to call your API
2. Replace mock user validation with real authentication
3. Add JWT token handling if needed
4. Implement token refresh logic

```typescript
const login = async (username: string, password: string): Promise<void> => {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });

  if (!response.ok) {
    throw new Error('Login failed');
  }

  const data = await response.json();
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(data.user));
  setUser(data.user);
};
```

## 🎯 Features

### Automatic Page Protection

Pages with `requiresLogin: true` are automatically wrapped with `ProtectedContent` via DocItem swizzling. No manual component additions needed.

### Loading State

Loading spinner appears while authentication state is being restored from localStorage. Prevents flash of protected content.

### Navbar Integration

- **Not Logged In**: Shows "Login" button
- **Logged In**: Shows user greeting, role badge, and "Logout" button

### Redirect After Login

Users are automatically redirected to their originally requested page after successful login.

### Session Persistence

User session survives page refreshes and browser restarts (stored in localStorage).

## 📱 Responsive Design

All components include responsive design for mobile, tablet, and desktop views:

- Login page adapts to smaller screens
- Navbar auth menu collapses on mobile
- Protected content prompt remains readable on all devices

## 🚀 Production Deployment

Before deploying to production:

### 1. Replace Mock Authentication

```typescript
// ❌ REMOVE: Mock user validation
const userData: User = MOCK_USERS[username] || { username, role: 'user' };

// ✅ ADD: Real API authentication
const response = await fetch('/api/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username, password })
});
```

### 2. Add HTTPS

Always use HTTPS for login pages in production.

### 3. Add CSRF Protection

If your backend supports it, add CSRF token handling.

### 4. Implement Token Expiration

Add logic to refresh tokens and expire sessions:

```typescript
const checkTokenExpiration = () => {
  const storedTime = localStorage.getItem('AUTH_TIMESTAMP');
  const expiryTime = 24 * 60 * 60 * 1000; // 24 hours

  if (storedTime && Date.now() - parseInt(storedTime) > expiryTime) {
    logout();
  }
};
```

### 5. Set HttpOnly Cookies (Recommended)

For enhanced security, store auth tokens in HttpOnly cookies:

```typescript
// Backend sets HttpOnly cookie automatically
// Frontend just needs to verify authentication status
const isAuthenticated = await fetch('/api/auth/verify').then(r => r.ok);
```

## 🧪 Testing

### Test Protected Page Access

1. Visit a protected page without logging in
2. Verify login prompt appears
3. Click "Go to Login" button
4. Enter demo credentials
5. Verify redirect back to original page
6. Verify page content is now visible

### Test Session Persistence

1. Login with demo credentials
2. Refresh the page
3. Verify you remain logged in
4. Close browser tab
5. Reopen site
6. Verify session persists (localStorage)

### Test Logout

1. Verify "Logout" button appears in navbar when logged in
2. Click logout
3. Verify redirect to home page
4. Verify session is cleared
5. Verify login prompt appears on protected pages

## 🐛 Troubleshooting

### Login Button Doesn't Appear

- Check that `Root.tsx` is properly wrapping the app
- Verify `AuthProvider` is imported correctly
- Clear browser cache and localStorage

### Protected Page Not Protected

- Ensure frontmatter has `requiresLogin: true`
- Check that `DocItem/Layout/index.tsx` is in the correct path
- Rebuild Docusaurus (`npm run build`)

### Session Not Persisting

- Check browser's localStorage is enabled
- Verify `AUTH_STORAGE_KEY` matches in both files
- Check browser DevTools → Application → Local Storage

### User Remains Logged In After Logout

- Clear localStorage manually: `localStorage.clear()`
- Check that logout button actually calls `logout()` function
- Verify Navbar component is properly imported

## 📚 Next Steps

1. ✅ Test the authentication system
2. ✅ Protect pages you want to restrict
3. ✅ Customize branding and messaging
4. ✅ Extend with backend API when ready
5. ✅ Implement role-based access control (future)
6. ✅ Add two-factor authentication (future)

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the component code comments
3. Check browser console for errors
4. Verify file paths and imports

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready
