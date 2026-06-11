# YOLO Intuition
## Detailed Voice Over Script
## Slides 7–11

---

# Slide 7: What Each Cell Predicts

Now that we understand how YOLO divides the image into a grid and assigns responsibility to specific cells, the next question becomes:

What exactly does a responsible cell predict?

Think about a grid cell that has been assigned a dog.

The cell now has a job to do.

It cannot simply say:

"There is a dog here."

That information alone is not enough.

The system must know:

- Where the dog is.
- How large the dog is.
- How confident the model is.
- What type of object it is.

Therefore, every grid cell produces two major categories of predictions.

The first category is bounding box predictions.

These describe the location and size of an object.

The second category is class probabilities.

These describe what type of object the model believes it has detected.

You can think of each grid cell as a mini object detector.

Every detector independently generates information about its region.

Some cells may predict nothing useful because no object exists there.

Other cells may generate strong predictions because they contain important objects.

When all grid cells make predictions simultaneously, the entire image can be analyzed in a single forward pass.

This is a major reason why YOLO is so fast.

The work is distributed across many cells, and every cell contributes information at the same time.

In the next slide, we will look more closely at the information contained inside a bounding box prediction.

---

# Slide 8: Bounding Box Predictions

Let's focus on the first type of prediction produced by a grid cell.

The bounding box.

Suppose a grid cell is responsible for detecting a dog.

The model must describe where the dog is located.

To do that, YOLO predicts five values.

These values are:

x

y

w

h

and confidence.

Let's understand each one.

The value x represents the horizontal position of the object's center.

The value y represents the vertical position of the object's center.

Together, x and y tell us where the center of the object is located.

Next, we have w and h.

The value w represents the width of the bounding box.

The value h represents the height of the bounding box.

Together, these four values define the location and size of the object.

Finally, we have confidence.

Confidence tells us how certain the model is that an object exists and how accurate the predicted box is.

You can think of these five values as a compact description of an object.

Whenever YOLO predicts an object, it uses these values to communicate:

Where the object is.

How large the object is.

And how confident it is about the prediction.

The remarkable thing is that every grid cell predicts these values simultaneously.

As a result, many objects can be detected in a single forward pass.

---

# Slide 9: Class Probabilities

Bounding boxes tell us where an object is.

But they do not tell us what the object is.

This is where class probabilities become important.

Suppose a grid cell predicts a bounding box.

The next question becomes:

What object is inside that box?

The network answers this question by predicting probabilities for all possible classes.

For example:

Dog

Cat

Car

Person

Bicycle

and many others.

Imagine the network produces the following probabilities:

Dog equals ninety-two percent.

Cat equals four percent.

Car equals two percent.

Person equals one percent.

Bicycle equals one percent.

What does this mean?

The model strongly believes the detected object is a dog.

The highest probability corresponds to the most likely class.

An important detail in YOLO Version 1 is that class probabilities are predicted per grid cell.

Not per bounding box.

This means all bounding boxes predicted by the same cell share the same class probability vector.

At this stage, YOLO now has two important pieces of information.

First, the location of the object through bounding boxes.

Second, the identity of the object through class probabilities.

Combining these two pieces allows the model to perform object detection.

Bounding boxes answer:

Where?

Class probabilities answer:

What?

Together they form the foundation of YOLO's predictions.

---

# Slide 10: Final Output Tensor

Now let's step back and look at the entire image.

Suppose we divide an image into a seven-by-seven grid.

That gives us forty-nine cells.

Every single cell produces predictions.

Each cell predicts:

Bounding box information.

Confidence values.

Class probabilities.

All these predictions must be stored somewhere.

The structure that stores these predictions is called the output tensor.

A tensor is simply a multidimensional collection of numbers.

For YOLO Version 1, the output shape is:

S by S by B times 5 plus C.

At first glance this looks complicated.

But it is actually straightforward.

S represents the grid size.

B represents the number of bounding boxes.

C represents the number of classes.

Let's use actual numbers.

Suppose:

S equals 7.

B equals 2.

C equals 80.

The output becomes:

Seven by seven by ninety.

What does ninety represent?

For each cell:

Two bounding boxes.

Each box requires five values.

That's ten numbers.

Then we add eighty class probabilities.

Ten plus eighty equals ninety.

So every cell produces ninety values.

Forty-nine cells produce forty-nine sets of ninety values.

This output tensor contains every prediction the network makes.

At this stage, YOLO has not yet produced final detections.

It has simply generated raw predictions.

These predictions will later be processed to create the final object detections we see on screen.

---

# Slide 11: Why This Works

Now that we understand the major components of YOLO, let's ask an important question.

Why does this approach work so well?

The first reason is the single forward pass.

Traditional detectors repeatedly analyzed different parts of an image.

YOLO processes the entire image once.

This dramatically reduces computation.

The second reason is that YOLO naturally handles multiple objects.

Different grid cells can detect different objects simultaneously.

A person can be detected in one cell.

A dog in another.

A bicycle in another.

Everything happens at the same time.

The third reason is global context.

Traditional detectors often looked at small cropped regions independently.

YOLO sees the entire image.

Because it sees the whole image, it can use contextual information.

For example, a bicycle is likely to appear near a person.

A car is likely to appear on a road.

This additional context helps improve predictions.

The fourth reason is speed.

Since everything is predicted in one pass, inference becomes extremely fast.

This makes YOLO suitable for real-time applications such as:

Autonomous vehicles.

Robotics.

Surveillance.

Industrial automation.

And mobile applications.

The real breakthrough of YOLO was not simply improving speed.

The breakthrough was changing how object detection was formulated.

Instead of searching for objects, YOLO predicts objects directly.

Instead of a multi-stage pipeline, YOLO uses a single end-to-end neural network.

This idea fundamentally changed computer vision and laid the foundation for many modern object detectors.