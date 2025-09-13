---
sidebar_position : 3
---
# Documents
A document is the basic unit of data in MongoDB.
It is similar to a row in a SQL database, but more flexible and stored in BSON format (Binary JSON).

Each document is a JSON-like structure made up of key-value pairs.

## Sample Document

```javascript
// document
{
    "name": "John",
    "age": 17,
    "class": "12A",
    "subjects": ["Math", "Physics", "Computer Science"]
}

```
Here, `"name"`, `"age"`, `"class"`, and `"subjects"` are keys (or fields)

Their values can be:
- Strings (`"John"`)
- Numbers (`17`)
- Arrays (`["Math", "Physics", "Computer Science"]`)
- Booleans, objects, dates, and more


### Analogy:
Imagine a document as a profile card of a person,

| Relation DB | MongoDB  |
|-------------|----------| 
| Column Name | Fields   |
| Values      | Records  | 

