# Gym management basics

A gym management system tracks three things: members, attendance, and
subscriptions. Members register with a name and contact details. Each
visit is logged as an attendance record with a date. A subscription
defines a plan, a start date, and an end date, and the system can check
whether a member's subscription is active on any given day.

Two derived views are useful. A daily attendance report counts visits
per day, which shows peak hours. A renewal report lists subscriptions
ending within the next seven days, so staff can follow up before a
member lapses.

Class scheduling adds a fourth piece: classes have an instructor, a
time slot, and a capacity. Booking reduces the remaining spots, and a
class with zero spots left should not accept new bookings.
