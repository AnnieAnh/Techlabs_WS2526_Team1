# Pipeline Quality Report

## Summary

- Total rows: **13187**
- Extraction coverage: **100.0%**

## Field Coverage

- `benefits`: ▓▓▓▓▓▓▓▓▓░ 90.1%
- `benefits_evidence`: ▓▓▓▓▓▓▓▓▓░ 90.1%
- `job_family`: ▓▓▓▓▓▓▓▓▓▓ 100.0%
- `job_summary`: ▓▓▓▓▓▓▓▓▓░ 99.9%
- `nice_to_h_have_skills`: ░░░░░░░░░░ 0.0%
- `nice_to_have_skills`: ▓▓▓▓░░░░░░ 46.9%
- `nice_to_have_skills_evidence`: ▓▓▓▓░░░░░░ 46.9%
- `soft_skills`: ▓▓▓▓▓▓▓▓▓░ 93.7%
- `tasks`: ▓▓▓▓▓▓▓▓▓░ 99.7%
- `tasks_evidence`: ▓▓▓▓▓▓▓▓▓░ 99.7%
- `technical_skills`: ▓▓▓▓▓▓▓▓▓░ 99.6%
- `technical_skills_evidence`: ▓▓▓▓▓▓▓▓▓░ 99.6%

## Categorical Distributions

### contract_type
- null: 12954
- Full-time: 3957
- Permanent: 1059
- Part-time: 822
- Working Student: 115
- Contract: 98
- Internship: 84
- Freelance: 59

### work_modality
- null: 7651
- Hybrid: 5978
- Remote: 4660
- On-site: 859

### seniority
- null: 13214
- Senior: 4220
- Lead: 896
- Junior: 751
- Director: 42
- C-Level: 25

## Top 20 Technical Skills

- Python: 2886
- CI/CD: 2499
- Azure: 2028
- AWS: 1902
- Java: 1861
- Kubernetes: 1853
- SQL: 1772
- Docker: 1678
- JavaScript: 1383
- Linux: 1368
- Git: 1309
- TypeScript: 1285
- GCP: 1195
- Software Development: 1123
- React: 1117
- Monitoring: 1040
- Terraform: 1003
- Software Architecture: 963
- DevOps: 962
- Microservices: 929

## Top 20 Soft Skills

- communication: 5487
- teamwork: 4373
- collaboration: 3100
- self-reliance: 2649
- analytical thinking: 2288
- problem-solving: 1626
- communication skills: 1325
- team spirit: 1321
- responsibility: 1240
- structured work approach: 952
- initiative: 949
- solution-oriented: 910
- leadership: 872
- curiosity: 801
- customer orientation: 786
- mentoring: 756
- willingness to learn: 717
- respectful interaction: 686
- creativity: 659
- independence: 654

## Salary Statistics

- Rows with salary: 0 (0.0%)

## Top 10 Cities

- Berlin: 1947
- Hamburg: 1076
- München: 946
- Frankfurt am Main: 659
- Stuttgart: 547
- Düsseldorf: 476
- Dresden: 446
- Munich: 436
- Leipzig: 410
- Köln: 354

## Top 10 States

- North Rhine-Westphalia: 3568
- Bavaria: 3126
- Baden-Württemberg: 2788
- Berlin: 1995
- Hesse: 1552
- Hamburg: 1082
- Saxony: 994
- Lower Saxony: 758
- Schleswig-Holstein: 415
- Rhineland-Palatinate: 371

## Validation Summary

- Clean rows: 11591 (87.9%)
- Rows with flags: 1596

- `skill_not_in_description`: 756
- `remote_but_onsite_text`: 567
- `list_truncated`: 263
- `intern_contract_mismatch`: 126
- `no_skills_long_description`: 39
- `high_hallucination_rate`: 29
- `experience_unrealistic`: 25
- `below_floor`: 6

## Hallucination Summary

- Total skill-not-in-description flags: 756
- High-hallucination rows: 29

- 'Network Administration': 61
- 'AWS': 27
- 'C#': 19
- 'ABSL': 17
- 'Software Development': 11
- 'Software Architecture': 11
- 'Data Management': 11
- 'Data Analysis': 10
- 'GCP': 10
- 'Data Quality': 10

## Benefit Coverage

- Rows with benefits: 90.1%
- Rows with benefit categories: 0.0%

## Quality Concerns

- nice_to_h_have_skills: 0.0% coverage (below 50% threshold)
- nice_to_have_skills: 46.9% coverage (below 50% threshold)
- nice_to_have_skills_evidence: 46.9% coverage (below 50% threshold)
