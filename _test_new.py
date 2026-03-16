import sys
sys.path.insert(0, '.')
from agent import _extract_credentials_from_task, _classify_task, _parse_task_constraints

# Test 1: trailing space username
creds = _extract_credentials_from_task("Please log in using username equals 'user ' and password equals 'Passw0rd!'.")
print('LOGIN creds:', creds)
assert creds.get('username') == 'user ', f'Expected "user " (with space), got {repr(creds.get("username"))}'
assert creds.get('password') == 'Passw0rd!', f'Got password={repr(creds.get("password"))}'
print('TEST 1 PASS: trailing space preserved')

# Test 2: CONTAINS job title
creds2 = _extract_credentials_from_task("User initiates job posting by writing a strong title of the job where the query CONTAINS 'opers J'")
print('JOB TITLE CONTAINS creds:', creds2)
print('TEST 2: job_title_contains =', repr(creds2.get('job_title_contains')))

# Test 3: classify new task types
tests = [
    ("Update quantity of item with title 'Adidas Tiro Training Pants' in my cart to '9'", "QUANTITY_CHANGED"),
    ("Increase the quantity of the item in the cart to 5.", "ITEM_INCREMENTED"),
    ("Show details for a product where the brand equals 'Apple' and the price less_than 1000", "VIEW_DETAIL"),
    ("Mark as spam the email with subject that CONTAINS 'ted Ti' from an email address that does NOT CONTAIN 'bsr'", "MARK_AS_SPAM"),
    ("Delete task whose name not contains 'hpc' and description contains 'cation' and date less than '2026-03-01' and priority equals '1'.", "AUTOLIST_DELETE_TASK"),
    ("User initiates a process of job posting by writing a title of job for 'Data Scientists Jobs'", "WRITE_JOB_TITLE"),
    ("Switch to day view in the calendar.", "SELECT_DAY"),
    ("Open the event creation wizard to add an event with a title that contains 'oppi'.", "EVENT_WIZARD_OPEN"),
    ("Click on cell for a date in the 5 days view that is AFTER '2026-03-17 00:00:00'", "CELL_CLICKED"),
    ("Adjust the number of guests to 2 where guests_to is less than 3", "EDIT_NUMBER_OF_GUESTS"),
    ("Share the hotel listing with daniel_choi@webmail.net located in 'Prague'", "SHARE_HOTEL"),
    ("Show me details for popular hotels where the rating is greater than or equal to '4.5'", "POPULAR_HOTELS_VIEWED"),
    ("Mark the email as unread where from_email equals 'david.johnson@outlook.com'", "MARK_AS_UNREAD"),
    ("Change the application theme to 'dark'.", "THEME_CHANGED"),
    ("Save the post where the author CONTAINS 'Whit' and the content CONTAINS 'my day!'", "SAVE_POST"),
    ("Navigate to the 'Home' tab from the navbar.", "HOME_NAVBAR"),
    ("Please collapse the menu for the restaurant where the rating is NOT '5.5'", "COLLAPSE_MENU"),
    ("Show me my hidden posts.", "VIEW_HIDDEN_POSTS"),
    ("Search for jobs where the query does NOT CONTAIN 'eld'.", "SEARCH_JOBS"),
    ("Change the priority to 'High' where the priority is NOT 'Low'.", "AUTOLIST_SELECT_TASK_PRIORITY"),
    ("Cancel creating the task where the name equals 'Implement rate limiting'", "AUTOLIST_CANCEL_TASK_CREATION"),
    ("Create a team whose name contains 'u', description equals 'Manages recruitment'", "AUTOLIST_TEAM_CREATED"),
    ("Search ride details where the location is 'Portland State School'", "SEARCH_RIDE"),
    ("Enter and select a location where the location equals 'City Boutique'", "ENTER_LOCATION"),
    ("Contact a doctor where doctor_name equals 'Dr. Susan Moore' and patient_name equals 'Ava'", "DOCTOR_CONTACTED_SUCCESSFULLY"),
    ("Show me the availability details for a doctor where the doctor_name is NOT 'Dr. Thomas'", "VIEW_DOCTOR_AVAILABILITY"),
    ("Submit a review for a restaurant where name contains 'o' and author equals 'James'", "REVIEW_SUBMITTED"),
    ("Return to all restaurants where the name equals 'Pho Saigon'", "BACK_TO_ALL_RESTAURANTS"),
    ("Navigate to the Help page to find guidance, FAQs, or troubleshooting information.", "HELP_PAGE_VIEW"),
    ("Show details for the help category where the category equals 'Reservations'", "HELP_CATEGORY_SELECTED"),
    ("Search for matters where the query does NOT contain 'Regulatory Approval'.", "SEARCH_MATTER"),
    ("Show me details for clients whose status is NOT 'On Hold'", "FILTER_CLIENTS"),
]
all_pass = True
for prompt, expected in tests:
    got = _classify_task(prompt)
    if got != expected:
        print(f'FAIL: expected {expected}, got {got} for: {prompt[:60]}')
        all_pass = False
    else:
        print(f'OK: {expected}')

# Test 4: REGISTRATION with trailing space
creds4 = _extract_credentials_from_task("Please register using username equals 'newuser ', email equals 'newuser @gmail.com' and password equals 'Passw0rd!'")
print('REGISTRATION creds:', creds4)
assert creds4.get('username') == 'newuser ', f'Expected "newuser " (with space), got {repr(creds4.get("username"))}'
print('TEST 4 PASS: newuser trailing space preserved')

if all_pass:
    print('\nALL CLASSIFIER TESTS PASSED')
else:
    print('\nSOME TESTS FAILED')
