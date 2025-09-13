---
sidebar_position: 2
---

# Create Collection

## What is a Collection in MongoDB?
A collection is like a table in SQL, but more flexible.
It holds documents (which are like rows), and all documents are stored in BSON format (Binary JSON).

Example: In a school database, you might have collections like:
- students
- teachers
- classes

✅ A collection is a group of related documents.


## Analogy

| Concept | Relational DB | MongoDB    | Analogy                             |
| ------- | ------------- | ---------- | ----------------------------------- |
| Top     | Database      | Database   | A folder                            |
| Middle  | Table         | Collection | A spreadsheet inside the folder     |
| Bottom  | Row           | Document   | A single record (e.g., one student) |



## Creating Collection

### Using MongoDB Shell

```javascript
// Switch to the database - bookstore
use bookstore

// Create a collection (collections are created automatically when you insert data)
db.createCollection("books")
```

### View all collection in database
```javascript
// View collections in current database
show collections
```

### Using MongoDB Compass

1. Open MongoDB Compass
2. Connect to your MongoDB instance
3. Select the database `bookstore`
4. Enter collection name: `books`
5. Click "Create Database"

