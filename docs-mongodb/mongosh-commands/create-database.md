---
sidebar_position: 1
---

# Create Database

Welcome to MongoDB Fundamentals! This comprehensive guide will take you through essential MongoDB operations using both MongoDB Shell and MongoDB Compass. By the end of this tutorial, you'll have a solid foundation in database operations, document manipulation, and query techniques.

## What is Database in MongoDB ?
What is a Database in MongoDB?
A database in MongoDB is like a folder that holds your data.
Each database contains collections, which are like tables in relational databases. 
- Example: You could have a database called school, ecommerce, or hospital.

✅ Think of a database as the top-level container for data.

## Getting Started

Before we begin, ensure you have MongoDB installed on your system or access to MongoDB Atlas. You can use either:
- **MongoDB Shell** (mongosh) - Command-line interface
- **MongoDB Compass** - Graphical user interface

## Creating Database

### Using MongoDB Shell

```javascript
// Create a database
use bookstore

// Create another database
use bookstall
```

### Switch between Database
```javascript
// Same command `use` to swtich to database
use bookstore

// View all databases
show dbs
```

## Using MongoDB Compass

1. Open MongoDB Compass
2. Connect to your MongoDB instance
3. Click "Create Database"
4. Enter database name: `bookstore`
