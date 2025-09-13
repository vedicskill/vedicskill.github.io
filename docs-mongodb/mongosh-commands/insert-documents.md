---
sidebar-position: 4
---

# Insert One Document
MongoDB is a document-based NoSQL database that stores data in flexible, JSON-like structures called documents. When working with MongoDB, one of the most common operations is inserting a single document into a collection using the insertOne() method.


## insertOne()
The `db.collection.insertOne()` method allows you to insert a single document into a MongoDB collection. If the collection doesn’t already exist, MongoDB will create it automatically when the document is inserted.

```javascript
// Insert single document
db.collection_name.insertOne({
key1: value1,
key2: value2,
...
})
```

- `db`: refers to the current database you're using
- `collection`: the target collection 
- The `{}` block is the document you want to insert

✅ Example: Insert One Student Record

Suppose you have a database called school and you want to store student details in a collection named students. Here's how you insert one student:

### Syntax for inserting documents into collection

```javascript
use school  // switch to the 'school' database

// create collection
db.createCollection("students")

db.students.insertOne({
name: "John",
age: 17,
class: "12A",
subjects: ["Math", "Science"]
})
```

- This creates a new document in the `students` collection.
- If `students` does not exist, MongoDB will create it automatically.
- The `subjects` field stores an array of strings.

🔐 Auto-generated `_id` Field
When a document is inserted, MongoDB automatically adds a unique _id field to identify it:

```json
{
"_id": ObjectId("66bfabcd1234567890efabcd"),
"name": "Sudhir",
"age": 17,
"class": "12A",
"subjects": ["Math", "Science"]
}
```

You can also manually set your own `_id` if needed.

### Key Benefits
- No need to predefine schema
- Collections are auto-created
- Flexible and fast for rapid development

## Conclusion
The `db.collection_name.insertOne()` method is a simple yet powerful way to add individual records to a MongoDB collection. It’s ideal when you're adding single users, products, orders, or any other unit of data. Mastering this method is the first step in working effectively with MongoDB.

