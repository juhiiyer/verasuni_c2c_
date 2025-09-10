CREATE TABLE colleges (
    college_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE academic_learning (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    college_id INT,
    subcategory VARCHAR(100),
    f1 TEXT,
    f2 TEXT,
    f3 TEXT,
    f4 TEXT,
    f5 TEXT,
    FOREIGN KEY (college_id) REFERENCES colleges(college_id)
);

INSERT INTO colleges (name) VALUES
('College 1'),
('College 2');

INSERT INTO academic_learning (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(1, 'Faculty Quality', '70% of faculty hold PhDs', '25% with industry background', 'Average teaching experience 12 years', 'Faculty-student mentoring ratio 1:25', '14 visiting professors annually'),
(1, 'Curriculum Flexibility', 'CBCS system operational since 2018', '65 electives across departments', '12 interdisciplinary courses per semester', 'Up to 20% credits transferable', NULL),
(1, 'Research Opportunities', '42 funded projects in last 3 years', '18 patents filed', '55% undergraduates in research', '3 specialized research centers', 'Annual research grant allocation ₹12 crore'),
(1, 'Industry Connections', '15 MoUs with companies', '82% students secure internships', 'Average stipend ₹22,000/month', '32 guest lectures annually', '10 industry-sponsored labs'),
(1, 'Library & Resources', '80,000 physical books', '120 subscribed journals', 'IEEE/ACM/JSTOR access', '1,200 seating capacity', 'Library open 8 AM–12 midnight'),
(1, 'Student-Faculty Ratio', 'Overall 1:18', 'CS: 1:14', 'Mechanical: 1:22', NULL, NULL),
(1, 'Classroom Infrastructure', '70% classrooms with smartboards', '40 projectors installed', '20 advanced laboratories', 'Lab-to-student ratio 1:28', '6 computer centers with 1,200 systems'),
(1, 'Assessment & Exams', 'Two mid-sems + one end-sem per semester', 'Continuous assessment 30%', 'Pass percentage 89%', 'Exam duration 3 hours', NULL);

INSERT INTO academic_learning (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(2, 'Faculty Quality', '52% of faculty hold PhDs', '15% with international exposure', 'Average teaching experience 9 years', 'Annual faculty training program', 'Faculty-student mentoring ratio 1:30'),
(2, 'Curriculum Flexibility', 'Curriculum revised every 5 years', '28 electives available', '5 interdisciplinary courses', 'Credit transfer up to 12%', 'Dual-degree option CS + Business'),
(2, 'Research Opportunities', '36 funded projects in last 3 years', '12 patents filed', '48% undergraduates engaged in research', '2 dedicated research centers', 'Annual grant allocation ₹8 crore'),
(2, 'Industry Connections', '10 MoUs with companies', '64% students secure internships', 'Average stipend ₹18,500/month', '20 guest lectures annually', '6 industry-backed labs'),
(2, 'Library & Resources', '50,000 physical books', '95 subscribed journals', 'Scopus/Springer/Wiley access', '850 seating capacity', 'RFID-enabled book borrowing'),
(2, 'Student-Faculty Ratio', 'Overall 1:22', 'Civil Eng: 1:18', 'Electrical Eng: 1:27', NULL, NULL),
(2, 'Classroom Infrastructure', '65% classrooms with projectors', '18 advanced labs', '4 seminar halls with AV', '800 workstations in computing labs', 'Lab-to-student ratio 1:32'),
(2, 'Assessment & Exams', '3 internal assessments + one end-sem', 'Internal weightage 40%', 'Pass percentage 86%', 'Exam sessions in 2 slots daily', NULL);


CREATE TABLE campus_life (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    college_id INT,
    subcategory VARCHAR(100),
    f1 TEXT,
    f2 TEXT,
    f3 TEXT,
    f4 TEXT,
    f5 TEXT,
    FOREIGN KEY (college_id) REFERENCES colleges(college_id)
);

INSERT INTO campus_life (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(1, 'Clubs & Societies', '48 active clubs (coding, drama, robotics, debating, photography)', 'Annual recruitment drives', 'Average 1,200 student participants annually', '4 inter-college fests host club competitions', '2 clubs funded by corporate sponsorships'),
(1, 'Sports & Athletics', '12 outdoor fields (football, cricket, hockey)', '6 indoor courts (badminton, basketball)', 'Annual inter-department sports meet with 2,000 participants', 'Fitness center open 6 AM–10 PM', '3 coaches + 2 physiotherapists on call'),
(1, 'Cultural Events', '2 annual cultural festivals with ~8,000 attendees', '15 regional language clubs perform annually', '3 state-level recognitions for performing arts', 'Music rooms with 30+ instruments', 'Traditional day organized twice per year'),
(1, 'Student Government', 'Elected council of 32 members', 'Elections every 12 months (~70% turnout)', 'Committees for finance, culture, sports, welfare', '2 faculty advisors', 'Annual budget of ₹1.5 crore'),
(1, 'Volunteering & Outreach', '8 NGOs partnered with university', '1,200 students in blood donation camps annually', 'NSS/NCC units with 500+ cadets', 'Rural development projects in 12 villages', '6-week summer outreach program with international volunteers');

INSERT INTO campus_life (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(2, 'Clubs & Societies', '26 active clubs (coding, entrepreneurship, dramatics, environment)', '~700 annual active members', 'Annual club showcase with 40 stalls', '1 entrepreneurship club incubated 3 startups', NULL),
(2, 'Sports & Athletics', '8 outdoor fields (cricket, football, athletics track)', '4 indoor facilities', 'Annual sports fest with 1,200 participants', 'Fitness center with 200 equipment units', '2 physiotherapists part-time'),
(2, 'Cultural Events', '1 major cultural fest with 5,000+ attendees', 'Annual talent night', '8 music & dance clubs', '2 student films screened at state competitions', 'Auditorium with 1,500 seating capacity'),
(2, 'Student Government', '18-member elected council', 'Elections every 18 months (~60% turnout)', '4 committees (welfare, cultural, academic, finance)', 'Annual budget ₹85 lakh', NULL),
(2, 'Volunteering & Outreach', 'Partnered with 5 NGOs', '~600 students involved yearly', 'Tree plantation drives (10,000 saplings in 5 years)', 'Rural literacy project in 6 schools', 'NCC unit with 150 cadets');



CREATE TABLE facilities (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    college_id INT,
    subcategory VARCHAR(100),
    f1 TEXT,
    f2 TEXT,
    f3 TEXT,
    f4 TEXT,
    f5 TEXT,
    FOREIGN KEY (college_id) REFERENCES colleges(college_id)
);

INSERT INTO facilities (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(1, 'Hostels & Housing', '14 hostels (10 male, 4 female), capacity 6,200', 'Single/double occupancy rooms', 'Wi-Fi across 90% hostel area', 'Biometric entry + 24/7 security', 'Mess with 4-week rotating menu'),
(1, 'Dining & Food Options', '4 student mess halls seating 1,500 each', '12 food kiosks on campus', 'Average ₹45 per meal subsidy', '3 international cuisine counters', 'Central kitchen produces 12,000 meals/day'),
(1, 'Healthcare Facilities', 'On-campus 40-bed medical center', '8 full-time doctors + 12 nurses', 'Tie-ups with 2 nearby hospitals', '3 ambulances available 24/7', 'Annual health checkups mandatory'),
(1, 'IT & Digital Access', 'Campus-wide 1 Gbps internet backbone', '2,400 access points', 'VPN access for research', '20 computer labs with 1,800 PCs', 'Digital ID used for attendance & library'),
(1, 'Transport Services', '32 buses on 18 city routes', '12,000 daily student commuters', 'GPS tracking installed', 'Subsidized fares (40% lower)', 'Cycle rental with 400 bicycles');

INSERT INTO facilities (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(2, 'Hostels & Housing', '8 hostels (5 male, 3 female), capacity 3,400', '70% double occupancy, 30% single', 'Wi-Fi enabled common rooms', '2 dining halls with 800 capacity each', 'Wardens appointed per block'),
(2, 'Dining & Food Options', '2 central mess halls', '6 campus canteens', 'Meals average ₹55', '2 vendor-run cafes', 'Menu rotates every 10 days'),
(2, 'Healthcare Facilities', '20-bed medical center', '5 doctors (2 specialists) + 6 nurses', 'Tie-up with 1 tertiary hospital', '2 ambulances (1 ICU-equipped)', 'Vaccination drives held annually'),
(2, 'IT & Digital Access', '600 Mbps internet backbone', '900 access points', '10 computer labs with 1,000 PCs', 'Biometric attendance for classes', 'Partial VPN access (faculty & postgrads)'),
(2, 'Transport Services', '18 buses on 12 routes', '~7,000 daily student commuters', 'GPS-enabled buses', 'Monthly pass scheme', '200 campus cycles maintained');


CREATE TABLE support_services (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    college_id INT,
    subcategory VARCHAR(100),
    f1 TEXT,
    f2 TEXT,
    f3 TEXT,
    f4 TEXT,
    f5 TEXT,
    FOREIGN KEY (college_id) REFERENCES colleges(college_id)
);

INSERT INTO support_services (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(1, 'Career Services', 'Dedicated career center with 20 staff members', '90+ companies recruit annually', 'Placement portal updated daily', 'Average placement rate 87%', 'Career counseling workshops every semester'),
(1, 'Counseling & Mental Health', 'On-campus counseling center with 5 licensed psychologists', 'Weekly therapy sessions available', 'Peer support groups organized monthly', 'Confidential helpline operational 24/7', 'Workshops on stress/time management'),
(1, 'Financial Aid', 'Merit and need-based scholarships available', '4,000 students receive aid yearly', '₹15 crore allocated annually', 'Loan facilitation via 3 partner banks', 'Emergency fund for low-income students'),
(1, 'International Student Office', 'Dedicated cell for 200+ foreign students', 'Orientation programs held bi-annually', 'Visa & documentation support', 'Language tutoring for non-English speakers', '24/7 international student helpline'),
(1, 'Disability Support', 'Assistive tech (screen readers, hearing devices)', 'Reserved seating in lecture halls', 'Accessible ramps and elevators in 80% buildings', 'Dedicated support staff for mobility aid', 'Extended exam timings for eligible students');

INSERT INTO support_services (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(2, 'Career Services', 'Career cell with 12 staff members', '60+ companies recruit annually', 'Placement portal updated weekly', 'Placement rate ~78%', 'Resume/Interview workshops held quarterly'),
(2, 'Counseling & Mental Health', '3 full-time counselors', 'Weekly group counseling sessions', 'Stress relief workshops every semester', '24/7 online mental health chatbot', NULL),
(2, 'Financial Aid', '2,200 students receive aid yearly', '₹9 crore allocated annually', 'Merit scholarships for top 5% students', 'MoU with 2 banks for education loans', NULL),
(2, 'International Student Office', 'Supports 80+ international students', 'Orientation week every semester', 'Assistance with FRRO registration', 'Cultural buddy program with locals', NULL),
(2, 'Disability Support', 'Wheelchair accessible classrooms (60%)', 'Note-taking support volunteers', 'Reserved hostel rooms near ground floor', 'Dedicated transportation service', NULL);


CREATE TABLE infrastructure (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    college_id INT,
    subcategory VARCHAR(100),
    f1 TEXT,
    f2 TEXT,
    f3 TEXT,
    f4 TEXT,
    f5 TEXT,
    FOREIGN KEY (college_id) REFERENCES colleges(college_id)
);

INSERT INTO infrastructure (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(1, 'Classrooms', '240 lecture halls total', '80% equipped with smartboards', 'Centralized AC in 70% buildings', 'Average class size 60 students', 'Timetable digitized via central portal'),
(1, 'Laboratories', '65 specialized labs (engineering, sciences)', 'Industry-sponsored labs (10 units)', '24/7 access for research scholars', 'Lab-to-student ratio 1:25', 'Annual equipment budget ₹6 crore'),
(1, 'Library', '80,000 physical volumes', '120 journals subscribed', 'Digital access to IEEE, ACM, JSTOR', 'Seating capacity 1,200', 'RFID book checkout system'),
(1, 'IT Infrastructure', 'Campus backbone 1 Gbps fiber', '2,400 access points campus-wide', 'VPN for faculty & researchers', 'Biometric attendance tracking', 'Cloud-based LMS integrated'),
(1, 'Green Infrastructure', '12 acres of green cover maintained', 'Rainwater harvesting system', 'Solar panels power 20% electricity', 'Zero-plastic policy since 2019', 'Waste segregation at source');

INSERT INTO infrastructure (college_id, subcategory, f1, f2, f3, f4, f5) VALUES
(2, 'Classrooms', '150 lecture halls total', '65% with projectors installed', 'Average class size 55 students', 'Centralized AC in 40% halls', 'Digital timetabling partially implemented'),
(2, 'Laboratories', '40 specialized labs (basic & applied sciences)', 'Industry-collab labs (5 units)', 'Lab-to-student ratio 1:30', 'Restricted access hours (8 AM–8 PM)', 'Annual equipment budget ₹3 crore'),
(2, 'Library', '50,000 physical volumes', '95 journals subscribed', 'Access to Scopus, Springer, Wiley', 'Seating capacity 850', 'Digital borrowing system via ID cards'),
(2, 'IT Infrastructure', 'Campus backbone 600 Mbps fiber', '900 access points', 'VPN for postgraduates only', 'Biometric attendance for some faculties', 'Online exam proctoring system in use'),
(2, 'Green Infrastructure', '8 acres green cover', 'Solar panels supply 12% power', 'Rainwater harvesting for hostels', 'Recycling plant operational since 2020', 'Tree plantation drives annually');

