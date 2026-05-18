import random
import pandas as pd
career_profiles = {
    "Data Scientist": {
        "skills": ["Python", "SQL", "Statistics", "Data Visualization", "Machine Learning"],
        "personalities": ["INTJ", "INTP", "ISTJ"],
        "salary_range": (1000000, 2000000),
        "related": ["AI Engineer", "Bioinformatics Specialist"]
    },
    "AI Engineer": {
        "skills": ["Python", "TensorFlow", "Model Deployment", "Math", "Deep Learning"],
        "personalities": ["INTP", "INTJ", "ISTP"],
        "salary_range": (1200000, 2500000),
        "related": ["Data Scientist", "Robotics Engineer"]
    },
    "Cybersecurity Analyst": {
        "skills": ["Network Security", "Ethical Hacking", "Penetration Testing", "SIEM", "Compliance"],
        "personalities": ["ISTJ", "INTJ", "ESTJ"],
        "salary_range": (800000, 1800000),
        "related": ["Cloud Architect"]
    },
    "Full Stack Developer": {
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
        "personalities": ["ISTP", "INTJ", "ENTP"],
        "salary_range": (700000, 1600000),
        "related": ["Cloud Architect", "UI/UX Designer"]
    },
    "Cloud Architect": {
        "skills": ["AWS", "Docker", "Kubernetes", "DevOps", "System Design"],
        "personalities": ["ENTJ", "INTJ", "ESTJ"],
        "salary_range": (1500000, 3000000),
        "related": ["Full Stack Developer", "Cybersecurity Analyst"]
    },
    "UI/UX Designer": {
        "skills": ["Figma", "Design Thinking", "Creativity", "User Research", "Prototyping"],
        "personalities": ["INFP", "ENFP", "ISFP"],
        "salary_range": (600000, 1400000),
        "related": ["Learning Experience Designer"]
    },
    "Robotics Engineer": {
        "skills": ["C++", "Sensors", "Control Systems", "Mechanical Design", "Python"],
        "personalities": ["ISTP", "INTJ", "INTP"],
        "salary_range": (800000, 1800000),
        "related": ["AI Engineer"]
    },
    "Bioinformatics Specialist": {
        "skills": ["Biology", "Genomics", "R", "Data Analysis", "Python"],
        "personalities": ["INTP", "ISTJ", "INFJ"],
        "salary_range": (600000, 1500000),
        "related": ["Data Scientist"]
    },
    "Digital Marketing Strategist": {
        "skills": ["SEO", "Google Ads", "Content Marketing", "Analytics", "Email Marketing"],
        "personalities": ["ENFP", "ENTJ", "ESFP"],
        "salary_range": (600000, 1200000),
        "related": ["Product Manager"]
    },
    "Product Manager": {
        "skills": ["Roadmapping", "User Research", "Agile", "Communication", "Analytics"],
        "personalities": ["ENTJ", "ENTP", "ESTJ"],
        "salary_range": (1800000, 3500000),
        "related": ["Digital Marketing Strategist"]
    },
    "Technical Writer": {
        "skills": ["Writing", "Markdown", "APIs", "Technical Research", "Documentation"],
        "personalities": ["ISTJ", "INFJ", "ISFJ"],
        "salary_range": (500000, 1000000),
        "related": ["Learning Experience Designer"]
    },
    "Sustainability Consultant": {
        "skills": ["Environmental Science", "Policy", "Compliance", "Carbon Analysis", "Communication"],
        "personalities": ["INFJ", "ENFP", "ISFP"],
        "salary_range": (700000, 1500000),
        "related": ["Healthcare Operations Manager"]
    },
    "Financial Analyst": {
        "skills": ["Excel", "Accounting", "Financial Modeling", "Valuation", "Analytics"],
        "personalities": ["ISTJ", "INTJ", "ESTJ"],
        "salary_range": (600000, 1200000),
        "related": ["Product Manager"]
    },
    "Healthcare Operations Manager": {
        "skills": ["Administration", "Health Policy", "Compliance", "Leadership", "Team Management"],
        "personalities": ["ESTJ", "ESFJ", "ISFJ"],
        "salary_range": (800000, 1800000),
        "related": ["Sustainability Consultant"]
    },
    "Learning Experience Designer": {
        "skills": ["Instructional Design", "eLearning", "Curriculum Development", "Creativity", "Storyboarding"],
        "personalities": ["INFP", "ENFJ", "ISFJ"],
        "salary_range": (600000, 1100000),
        "related": ["UI/UX Designer", "Technical Writer"]
    }
}
all_personalities = sorted({p for c in career_profiles.values() for p in c["personalities"]})
data = []
for _ in range(3000):
    base_career = random.choice(list(career_profiles.keys()))
    base_skills = career_profiles[base_career]["skills"]
    related_careers = career_profiles[base_career]["related"]

    
    selected_skills = set(random.sample(base_skills, k=random.randint(3, 4)))
    for rel in related_careers:
        rel_skills = career_profiles[rel]["skills"]
        selected_skills.update(random.sample(rel_skills, k=random.randint(1, 2)))
        if len(selected_skills) >= 10:
            break
    selected_skills = list(selected_skills)[:10]

    
    selected_personalities = random.sample(all_personalities, k=random.randint(1, 3))

    
    scores = []
    for name, props in career_profiles.items():
        req_skills = set(props["skills"])
        skill_match = len(set(selected_skills) & req_skills)
        personality_match = len(set(selected_personalities) & set(props["personalities"]))
        score = skill_match * 2 + personality_match
        scores.append((name, score, skill_match, req_skills))

    scores.sort(key=lambda x: -x[1])
    best_career, _, best_match, best_skills = scores[0]
    alt_career, _, _, alt_skills = scores[1]

    
    missing_best = list(best_skills - set(selected_skills))
    missing_alt = list(alt_skills - set(selected_skills))

    
    min_sal, max_sal = career_profiles[best_career]["salary_range"]
    if best_match >= len(best_skills) - 1:
        salary = max_sal
    elif best_match <= 2:
        salary = min_sal
    else:
        ratio = best_match / len(best_skills)
        salary = int(min_sal + (max_sal - min_sal) * ratio)

    data.append({
        "selected_skills": ", ".join(selected_skills),
        "personality_types": ", ".join(selected_personalities),
        "expected_salary": salary,
        "career_match": best_career,
        "alt_career": alt_career,
        "missing_skills_best": ", ".join(missing_best),
        "missing_skills_alt": ", ".join(missing_alt)
    })
df = pd.DataFrame(data)
df.to_csv("valid_career_dataset.csv", index=False)
print("✅ Dataset saved as 'valid_career_dataset.csv'")
