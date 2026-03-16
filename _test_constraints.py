from agent import _parse_task_constraints, _format_constraints_block, _classify_task, _extract_credentials_from_task

# Test 1: destination
t1 = "Enter destination value that is NOT 'Business Tower - Atlanta, GA 30325, USA'."
c1 = _parse_task_constraints(t1)
print("T1 type:", _classify_task(t1))
print("T1 constraints:", c1)
print()

# Test 2: email spam
t2 = "Mark as spam the email with subject that CONTAINS 'ted Ti' from an email address that does NOT CONTAIN 'bsr' and is_spam EQUALS True."
c2 = _parse_task_constraints(t2)
print("T2 type:", _classify_task(t2))
print("T2 constraints:", c2)
print()

# Test 3: delete task
t3 = "Delete task whose name not contains 'hpc' and description contains 'cation' and date less than '2026-03-01' and priority equals '1'."
c3 = _parse_task_constraints(t3)
print("T3 type:", _classify_task(t3))
print("T3 constraints:", c3)
print("T3 creds:", _extract_credentials_from_task(t3))
print()

# Test 4: booking confirm
t4 = ("Please confirm the booking details for a stay where guests_set equals '1' AND "
      "host_name contains 'tor' AND reviews equals '2700' AND "
      "amenities is not one of ['High-speed Wifi', 'Great location', 'Parking'] AND "
      "title not equals 'The Ritz-Carlton Moscow' AND rating equals '4.9' AND "
      "price greater than '895' AND location contains 'It' AND "
      "card_number not equals '5500000000000004' AND expiration not equals '06/26' AND "
      "cvv equals '789' AND zipcode equals '12345' AND country equals 'Canada'")
c4 = _parse_task_constraints(t4)
print("T4 type:", _classify_task(t4))
print("T4 creds:", _extract_credentials_from_task(t4))
print("T4 constraints block:")
print(_format_constraints_block(c4))
print()

# Test 5: job posting
t5 = "User initiates a process of job posting by writing a title of job for 'Data Scientists Jobs'"
print("T5 type:", _classify_task(t5))
print("T5 creds:", _extract_credentials_from_task(t5))
