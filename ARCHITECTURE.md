# 🏗️ VedicSkill Authentication System - Architecture

## System Overview

```mermaid
graph TB
    subgraph "User Interface"
        Navbar["🔄 Navbar<br/>Auth Menu"]
        LoginPage["🔑 Login Page<br/>/login"]
        DocPage["📄 Doc Page<br/>with frontmatter"]
        ProtectedUI["🔒 Protected Content<br/>Prompt"]
    end

    subgraph "State Management"
        AuthContext["⚡ AuthContext<br/>React Context"]
        LocalStorage["💾 localStorage<br/>Persistence"]
    end

    subgraph "Components & Pages"
        Root["🎯 Root Wrapper<br/>src/theme/Root.tsx"]
        ProtectedContent["🛡️ ProtectedContent<br/>Wrapper Component"]
        DocItemLayout["📋 DocItem Layout<br/>Swizzled"]
        NavbarSwizzle["🎨 Navbar Swizzle<br/>Enhanced Navbar"]
    end

    subgraph "Utilities"
        Hooks["🪝 useAuth Hook<br/>useProtectedRoute Hook"]
        Utils["🔧 Utility Functions<br/>Auth helpers"]
    end

    subgraph "Data Flow"
        Frontmatter["📝 Frontmatter<br/>requiresLogin"]
        AuthState["👤 Auth State<br/>User object"]
    end

    User["👨 User"]
    
    %% User interactions
    User -->|"visits"| DocPage
    User -->|"clicks Login"| LoginPage
    User -->|"enters credentials"| AuthContext
    User -->|"sees navbar"| Navbar

    %% Component flow
    Root -->|"wraps app"| AuthContext
    AuthContext -->|"provides state"| Navbar
    AuthContext -->|"provides state"| ProtectedContent
    
    DocPage -->|"contains"| Frontmatter
    DocItemLayout -->|"reads"| Frontmatter
    DocItemLayout -->|"wraps with"| ProtectedContent
    ProtectedContent -->|"checks"| AuthContext
    ProtectedContent -->|"shows"| ProtectedUI
    
    LoginPage -->|"uses"| AuthContext
    Navbar -->|"uses"| AuthContext
    Navbar -->|"shows state"| AuthContext
    
    AuthContext -->|"persists to"| LocalStorage
    LocalStorage -->|"restores from"| AuthContext
    
    AuthContext -->|"accessed via"| Hooks
    Hooks -->|"uses"| Utils

    style User fill:#667eea
    style AuthContext fill:#764ba2
    style LocalStorage fill:#e74c3c
    style LoginPage fill:#3498db
    style ProtectedUI fill:#e67e22
    style Root fill:#2ecc71
