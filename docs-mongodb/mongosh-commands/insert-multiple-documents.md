---
sidebar-position: 5
---
# Insert Multiple Documents


The `insertMany()` method allows you to insert multiple documents in a single operation, which is more efficient than inserting documents one by one.

## Basic Multiple Document Insertion

```javascript
// Insert multiple student documents
db.students.insertMany([
    {
        firstName: "John",
        lastName: "Smith",
        age: 20,
        email: "john.smith@email.com",
        major: "Computer Science",
        gpa: 3.8,
        enrolled: true,
        enrollmentDate: new Date("2023-09-01"),
        courses: ["CS101", "MATH201", "ENG102"]
    },
    {
        firstName: "Emma",
        lastName: "Johnson",
        age: 19,
        email: "emma.johnson@email.com",
        major: "Mathematics",
        gpa: 3.9,
        enrolled: true,
        enrollmentDate: new Date("2023-09-01"),
        courses: ["MATH301", "STAT201", "CS101"]
    },
    {
        firstName: "Michael",
        lastName: "Brown",
        age: 21,
        email: "michael.brown@email.com",
        major: "Physics",
        gpa: 3.6,
        enrolled: true,
        enrollmentDate: new Date("2022-09-01"),
        courses: ["PHYS301", "MATH401", "CHEM201"]
    },
    {
        firstName: "Sarah",
        lastName: "Davis",
        age: 22,
        email: "sarah.davis@email.com",
        major: "Biology",
        gpa: 3.7,
        enrolled: false,
        enrollmentDate: new Date("2021-09-01"),
        courses: ["BIO401", "CHEM301", "STAT201"]
    }
])
```

## Response from insertMany()

When you run `insertMany()`, MongoDB returns a response like this:

```javascript
{
  acknowledged: true,
  insertedIds: {
    '0': ObjectId('65f1a2b3c4d5e6f7a8b9c0d1'),
    '1': ObjectId('65f1a2b3c4d5e6f7a8b9c0d2'),
    '2': ObjectId('65f1a2b3c4d5e6f7a8b9c0d3'),
    '3': ObjectId('65f1a2b3c4d5e6f7a8b9c0d4')
  }
}
```

## insertMany() Options

### Ordered vs Unordered Insertion

```javascript
// Ordered insertion (default) - stops on first error
db.students.insertMany([
    {firstName: "Alex", lastName: "Wilson", age: 20, major: "Engineering"},
    {firstName: "Lisa", lastName: "Taylor", age: 19, major: "Art"},
    {firstName: "David", lastName: "Miller", age: 21, major: "History"}
], {ordered: true})

// Unordered insertion - continues even if some documents fail
db.students.insertMany([
    {firstName: "Jessica", lastName: "Garcia", age: 20, major: "Psychology"},
    {firstName: "Ryan", lastName: "Martinez", age: 22, major: "Business"},
    {firstName: "Ashley", lastName: "Anderson", age: 19, major: "Chemistry"}
], {ordered: false})
```

### Write Concern Options

```javascript
// Insert with write concern
db.students.insertMany([
    {firstName: "Kevin", lastName: "Thomas", age: 20, major: "Economics"},
    {firstName: "Amanda", lastName: "Jackson", age: 21, major: "Literature"}
], {
    writeConcern: {
        w: "majority",
        j: true,
        wtimeout: 5000
    }
})
```

## Inserting Documents with Complex Data Types

### Documents with Nested Objects

```javascript
db.students.insertMany([
    {
        firstName: "Robert",
        lastName: "Lee",
        age: 20,
        email: "robert.lee@email.com",
        major: "Computer Science",
        gpa: 3.8,
        enrolled: true,
        personalInfo: {
            phone: "+1-555-0123",
            address: {
                street: "123 Main St",
                city: "Springfield",
                state: "IL",
                zipCode: "62701"
            },
            emergencyContact: {
                name: "Jane Lee",
                relationship: "Mother",
                phone: "+1-555-0124"
            }
        },
        academicInfo: {
            semester: "Fall 2024",
            year: "Junior",
            credits: 15,
            advisor: "Dr. Smith"
        },
        enrollmentDate: new Date("2022-09-01")
    },
    {
        firstName: "Maria",
        lastName: "Rodriguez",
        age: 19,
        email: "maria.rodriguez@email.com",
        major: "Biology",
        gpa: 3.9,
        enrolled: true,
        personalInfo: {
            phone: "+1-555-0125",
            address: {
                street: "456 Oak Ave",
                city: "Chicago",
                state: "IL",
                zipCode: "60601"
            },
            emergencyContact: {
                name: "Carlos Rodriguez",
                relationship: "Father",
                phone: "+1-555-0126"
            }
        },
        academicInfo: {
            semester: "Fall 2024",
            year: "Sophomore",
            credits: 16,
            advisor: "Dr. Johnson"
        },
        enrollmentDate: new Date("2023-09-01")
    }
])
```

### Documents with Arrays of Objects

```javascript
db.students.insertMany([
    {
        firstName: "Christopher",
        lastName: "White",
        age: 21,
        email: "chris.white@email.com",
        major: "Engineering",
        gpa: 3.7,
        enrolled: true,
        grades: [
            {
                course: "ENG201",
                courseName: "Thermodynamics",
                grade: "A",
                credits: 3,
                semester: "Spring 2024"
            },
            {
                course: "MATH301",
                courseName: "Calculus III",
                grade: "B+",
                credits: 4,
                semester: "Spring 2024"
            },
            {
                course: "PHYS201",
                courseName: "Physics II",
                grade: "A-",
                credits: 3,
                semester: "Fall 2023"
            }
        ],
        activities: [
            {
                name: "Engineering Club",
                role: "Vice President",
                startDate: new Date("2023-01-15"),
                active: true
            },
            {
                name: "Math Tutoring",
                role: "Tutor",
                startDate: new Date("2023-08-20"),
                active: true
            }
        ]
    },
    {
        firstName: "Jennifer",
        lastName: "Clark",
        age: 20,
        email: "jennifer.clark@email.com",
        major: "Psychology",
        gpa: 3.85,
        enrolled: true,
        grades: [
            {
                course: "PSY301",
                courseName: "Cognitive Psychology",
                grade: "A",
                credits: 3,
                semester: "Spring 2024"
            },
            {
                course: "STAT201",
                courseName: "Statistics",
                grade: "A-",
                credits: 3,
                semester: "Fall 2023"
            },
            {
                course: "BIO101",
                courseName: "General Biology",
                grade: "B+",
                credits: 4,
                semester: "Fall 2023"
            }
        ],
        activities: [
            {
                name: "Psychology Research Lab",
                role: "Research Assistant",
                startDate: new Date("2024-01-10"),
                active: true
            },
            {
                name: "Student Government",
                role: "Senator",
                startDate: new Date("2023-09-01"),
                active: true
            }
        ]
    }
])
```

## Inserting Documents with Different Data Types

```javascript
db.students.insertMany([
    {
        firstName: "Daniel",
        lastName: "Lewis",
        age: 22,
        email: "daniel.lewis@email.com",
        // String
        major: "Computer Science",
        // Number (Float)
        gpa: 3.75,
        // Boolean
        enrolled: true,
        // Date
        enrollmentDate: new Date("2021-09-01"),
        graduationDate: new Date("2025-05-15"),
        // Array
        skills: ["Python", "JavaScript", "Java", "SQL"],
        // Null
        internship: null,
        // Object
        scholarship: {
            name: "Merit Scholarship",
            amount: 5000,
            renewable: true
        }
    },
    {
        firstName: "Nicole",
        lastName: "Hall",
        age: 19,
        email: "nicole.hall@email.com",
        major: "Art",
        gpa: 3.6,
        enrolled: true,
        enrollmentDate: new Date("2023-09-01"),
        graduationDate: new Date("2027-05-15"),
        skills: ["Drawing", "Painting", "Digital Art", "Photography"],
        internship: {
            company: "Creative Agency",
            position: "Design Intern",
            startDate: new Date("2024-06-01"),
            endDate: new Date("2024-08-31"),
            paid: true
        },
        scholarship: null
    }
])
```
