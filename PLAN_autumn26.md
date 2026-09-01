# Course plan - Autumn 2026

## Contents

1. [Time and place](#time-and-place)
2. [Course material](#course-material)
3. [Lectures](#lectures)
4. [Exercise classes](#exercise-classes)
5. [DataCamp](#datacamp)
6. [Projects and exam](#projects-and-exam)

---

## Time and place

* **Lectures:** Wednesdays, 15:00-17:00, CSS 34-0-01, Chr. Hansen Auditorium
* **Exercise classes:** 3-hour sessions. Exact time and room depend on your assigned class.
* **Autumn break:** Week 42, no teaching
* **No in-person lecture on September 9:** structured self-study and DataCamp instead

## Course material

* **Lectures:** [NumEconCopenhagen/ProgEcon-lectures](https://github.com/NumEconCopenhagen/ProgEcon-lectures)
* **Exercises:** [NumEconCopenhagen/ProgEcon-exercises](https://github.com/NumEconCopenhagen/ProgEcon-exercises)
* Supplementary material and course announcements are available on Absalon.

---

## Lectures

| Week | Date | Lecture |
| --- | --- | --- |
| 36 | Wed, Sep 2 | **Lecture 1: Introduction** |
| 37 | Wed, Sep 9 | **No in-person lecture:** structured self-study, Getting Started and DataCamp |
| 38 | Wed, Sep 16 | **Lecture 2: Python Basics I**: variables, types, containers and conditionals |
| 39 | Wed, Sep 23 | **Lecture 3: Python Basics II**: loops, functions, methods, scope, copies and classes |
| 40 | Wed, Sep 30 | **Lecture 4: Numbers and NumPy**: floating point numbers and numerical arrays |
| 41 | Wed, Oct 7 | **Lecture 5: Printing and Plotting** |
| 42 | Wed, Oct 14 | *Autumn break* |
| 43 | Wed, Oct 21 | **Lecture 6: Optimization** |
| 44 | Wed, Oct 28 | **Lecture 7: Root-finding and Interpolation** |
| 45 | Wed, Nov 4 | **Lecture 8: Random Numbers and Simulation** |
| 46 | Wed, Nov 11 | **Lecture 9: Descriptive Economics I** |
| 47 | Wed, Nov 18 | **Lecture 10: Descriptive Economics II** + introduction to the data project |
| 48 | Wed, Nov 25 | **Lecture 11: Solow Model** |
| 49 | Wed, Dec 2 | **Lecture 12: Walras / Exchange Economy** + introduction to the model project and calibration |
| 50 | Wed, Dec 9 | **Lecture 13: Recap and Exam Preparation** |

### September 9

There is no in-person lecture on September 9. Lecture 2 follows on September 16.

Use the week to:

1. Make sure your Python installation and course setup work.
2. Work on Problem set 1: Introduction and getting started.
3. Complete *Introduction to Python*, Chapters 1-2, before the September 13 deadline.
4. If you finish early, look ahead at the comparison and boolean operators in *Intermediate Python*, Chapter 3. They are
   not required until October 4, but they make the conditionals in Lecture 2 easier.

---

## Exercise classes

Exercise classes run in **weeks 37-41 and 43-51**. Each class is three hours.

The early classes combine small programming problem sets with DataCamp. Later classes focus on numerical methods, data analysis, economic models and the two projects.

| Week | Main class activities |
| --- | --- |
| 37 | **Problem set 1: Introduction and getting started** - installation, notebooks + DataCamp |
| 38 | **Problem set 1** continued, coding practice + DataCamp |
| 39 | **Problem set 2: Python Basics I** + DataCamp |
| 40 | **Problem set 3: Python Basics II** + DataCamp |
| 41 | **Problem set 4: Numbers and NumPy** + DataCamp |
| 42 | *Autumn break* |
| 43 | **Problem set 5: Printing and Plotting** |
| 44 | **Problem set 6: Optimization** |
| 45 | **Problem set 7: Root-finding and Interpolation** |
| 46 | **Problem set 8: Random Numbers and Simulation** |
| 47 | **Problem set 9: Descriptive Economics I** |
| 48 | **Problem set 10: Descriptive Economics II** + data project workshop |
| 49 | **Problem set 11: Solow Model** + data project questions |
| 50 | **Problem set 12: Exchange Economy** + model project workshop |
| 51 | **Model project workshop + recap and exam preparation** |

The problem set number matches the numbered folder in the exercise repository, and both follow the lecture numbering.
Each exercise class works on the problem set for the *previous* week's lecture.

---

## DataCamp

DataCamp is used throughout the first part of the course to build basic Python skills. The assigned tasks should be completed continuously during the semester rather than immediately before the final deadline.

### Required courses

1. **Introduction to Python**
2. **Intermediate Python**

### Deadlines

All deadlines are at **23:59**.

| Deadline | Required progress |
| --- | --- |
| **Sun, Sep 13** | **Introduction to Python:** Chapter 1, *Python Basics*; Chapter 2, *Python Lists* |
| **Sun, Sep 20** | **Introduction to Python:** Chapter 3, *Functions and Packages*; Chapter 4, *NumPy* — **course completed** |
| **Sun, Sep 27** | **Intermediate Python:** Chapter 1, *Matplotlib*; Chapter 2, *Dictionaries & Pandas* |
| **Sun, Oct 4** | **Intermediate Python:** Chapter 3, *Logic, Control Flow and Filtering*; Chapter 4, *Loops* |
| **Sun, Oct 11** | **Intermediate Python:** Chapter 5, *Case Study: Hacker Statistics* — **course completed** |

Students are expected to complete the assigned DataCamp tasks by the respective deadlines.

### How DataCamp relates to the lectures

The two courses play different roles.

* ***Introduction to Python* is preparation.** Chapters 1-4 are due before the lecture that uses them, so you arrive
  with the vocabulary already in place.
* ***Intermediate Python* is consolidation.** Chapters 1-5 revisit and extend material you have already seen in the
  lecture and the exercise class.

| Lecture | Date | Related DataCamp material | Deadline | Role |
| --- | --- | --- | --- | --- |
| 2: Python Basics I | Sep 16 | *Introduction*, Ch. 1-2 | Sep 13 | Preparation |
| 2: Python Basics I - conditionals | Sep 16 | *Intermediate*, Ch. 3 | Oct 4 | Consolidation |
| 3: Python Basics II - loops, functions, scope, classes | Sep 23 | *Introduction*, Ch. 3; *Intermediate*, Ch. 4 | Sep 20 | Preparation |
| 4: Numbers and NumPy | Sep 30 | *Introduction*, Ch. 4 | Sep 20 | Preparation |
| 5: Printing and Plotting | Oct 7 | *Intermediate*, Ch. 1 | Sep 27 | Preparation |
| 8: Random Numbers and Simulation | Nov 4 | *Intermediate*, Ch. 5 | Oct 11 | Preview |
| 9-10: Descriptive Economics | Nov 11, 18 | *Intermediate*, Ch. 2 (pandas) | Sep 27 | Preview |

Two consequences worth knowing in advance:

* **Conditionals and loops are taught before DataCamp covers them.** Lecture 2 introduces conditionals and Lecture 3 introduces loops, both from scratch; the
  exercise class in week 40 is where you practise them, immediately before the October 4 deadline. Do not wait for
  DataCamp to explain them first.
* **Pandas is covered on DataCamp in September but not lectured until November.** Plan a short refresher of
  *Intermediate Python*, Chapter 2, before Lecture 9.

### Optional DataCamp material

Students who want additional practice can also work on:

1. Introduction to Functions in Python
2. Introduction to NumPy
3. Intermediate Object-Oriented Programming in Python

---

## Projects and exam

There are two mandatory projects during the semester:

1. **Data analysis project**
2. **Model analysis project**

Both projects must be handed in and approved.

| Activity | Deadline | Feedback |
| --- | --- | --- |
| DataCamp | According to the intermediate deadlines above | |
| Project 1: Data project | **Sun, Nov 29, 23:59** | Feedback before the end of teaching |
| Project 2: Model project | **Sun, Dec 13, 23:59** | Feedback before the exam |
| Exam | See Digital Exam | |

Projects and the exam can be completed individually or in groups of up to four students. **Groups may consist of students from different exercise classes ("holds").**


The model project includes applying numerical methods and calibrating an economic model. The data and model projects can be revised using feedback received during the semester before they are submitted as part of the exam portfolio.
