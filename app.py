from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)
model = joblib.load("xgboost_career_model.pkl")
mlb_skills = joblib.load("mlb_skills.pkl")
mlb_personality = joblib.load("mlb_personality.pkl")
le = joblib.load("label_encoder.pkl")

career_profiles = {
    "Data Scientist": {
        "skills": ["Python", "SQL", "Statistics", "Data Analysis", "Machine Learning"]
    },
    "AI Engineer": {
        "skills": ["Python", "TensorFlow", "Deep Learning", "Math", "Model Deployment"]
    },
    "Cybersecurity Analyst": {
        "skills": ["Network Security", "Ethical Hacking", "Penetration Testing", "SIEM", "Compliance"]
    },
    "Full Stack Developer": {
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js"]
    },
    "Cloud Architect": {
        "skills": ["AWS", "Docker", "Kubernetes", "DevOps", "System Design"]
    },
    "UI/UX Designer": {
        "skills": ["Figma", "Design Thinking", "Creativity", "User Research", "Prototyping"]
    },
    "Robotics Engineer": {
        "skills": ["C++", "Sensors", "Control Systems", "Python", "Mechanical Design"]
    },
    "Bioinformatics Specialist": {
        "skills": ["Biology", "Genomics", "Python", "R", "Data Analysis"]
    },
    "Digital Marketing Strategist": {
        "skills": ["SEO", "Google Ads", "Content Marketing", "Analytics", "Email Marketing"]
    },
    "Product Manager": {
        "skills": ["Roadmapping", "User Research", "Agile", "Communication", "System Design"]
    },
    "Technical Writer": {
        "skills": ["Writing", "Markdown", "APIs", "Technical Research", "Documentation"]
    },
    "Sustainability Consultant": {
        "skills": ["Environmental Science", "Policy", "Carbon Analysis", "Data Analysis", "Communication"]
    },
    "Financial Analyst": {
        "skills": ["Excel", "Accounting", "Financial Modeling", "Valuation", "Analytics"]
    },
    "Healthcare Operations Manager": {
        "skills": ["Administration", "Health Policy", "Leadership", "Data Analysis", "Team Management"]
    },
    "Learning Experience Designer": {
        "skills": ["Instructional Design", "eLearning", "Curriculum Development", "Storyboarding", "Communication"]
    }
}
youtube_links = {
    "Python": "https://www.youtube.com/watch?v=_uQrJ0TkZlc",
    "SQL": "https://www.youtube.com/watch?v=27axs9dO7AE",
    "Statistics": "https://www.youtube.com/watch?v=xxpc-HPKN28",
    "Data Visualization": "https://www.youtube.com/watch?v=AdSZJzb-aX8",
    "Machine Learning": "https://www.youtube.com/watch?v=Gv9_4yMHFhI",
    "TensorFlow": "https://www.youtube.com/watch?v=tPYj3fFJGjk",
    "Model Deployment": "https://www.youtube.com/watch?v=pYhPQoP3Y4o",
    "Math": "https://www.youtube.com/watch?v=OmJ-4B-mS-Y",
    "Deep Learning": "https://www.youtube.com/watch?v=aircAruvnKk",
    "Network Security": "https://www.youtube.com/watch?v=3QhU9jd03a0",
    "Ethical Hacking": "https://www.youtube.com/watch?v=3Kq1MIfTWCE",
    "Penetration Testing": "https://www.youtube.com/watch?v=5qw2M4scz2I",
    "SIEM": "https://www.youtube.com/watch?v=r8L39uw9v0Y",
    "Compliance": "https://www.youtube.com/watch?v=Hh0yXtVCR7g",
    "HTML": "https://www.youtube.com/watch?v=pQN-pnXPaVg",
    "CSS": "https://www.youtube.com/watch?v=yfoY53QXEnI",
    "JavaScript": "https://www.youtube.com/watch?v=W6NZfCO5SIk",
    "React": "https://www.youtube.com/watch?v=bMknfKXIFA8",
    "Node.js": "https://www.youtube.com/watch?v=TlB_eWDSMt4",
    "AWS": "https://www.youtube.com/watch?v=ulprqHHWlng",
    "Docker": "https://www.youtube.com/watch?v=fqMOX6JJhGo",
    "Kubernetes": "https://www.youtube.com/watch?v=X48VuDVv0do",
    "DevOps": "https://www.youtube.com/watch?v=j5Zsa_eOXeY",
    "System Design": "https://www.youtube.com/watch?v=xpDnVSmNFX0",
    "Figma": "https://www.youtube.com/watch?v=FTFaQWZBqQ8",
    "Design Thinking": "https://www.youtube.com/watch?v=a7sEoEvT8l8",
    "Creativity": "https://www.youtube.com/watch?v=f7FVZlGGLFg",
    "User Research": "https://www.youtube.com/watch?v=5b8nL1xdZzI",
    "Prototyping": "https://www.youtube.com/watch?v=1nDBbYwhklo",
    "C++": "https://www.youtube.com/watch?v=vLnPwxZdW4Y",
    "Sensors": "https://www.youtube.com/watch?v=NeVb8p-MBOI",
    "Control Systems": "https://www.youtube.com/watch?v=GPOgZ3m2lqI",
    "Mechanical Design": "https://www.youtube.com/watch?v=ccq7Gxt4xuM",
    "Biology": "https://www.youtube.com/watch?v=5tjcZ3k4kzA",
    "Genomics": "https://www.youtube.com/watch?v=1fiJupfbSpg",
    "R": "https://www.youtube.com/watch?v=_V8eKsto3Ug",
    "Data Analysis": "https://www.youtube.com/watch?v=r-uOLxNrNk8",
    "SEO": "https://www.youtube.com/watch?v=E97n6pcUJlY",
    "Google Ads": "https://www.youtube.com/watch?v=ME9bq_5dZ5E",
    "Content Marketing": "https://www.youtube.com/watch?v=z_l3DLUFGqU",
    "Analytics": "https://www.youtube.com/watch?v=4UccH9eXybo",
    "Email Marketing": "https://www.youtube.com/watch?v=HQuvx5FfE6A",
    "Roadmapping": "https://www.youtube.com/watch?v=nTB2zDrlXj8",
    "User Research": "https://www.youtube.com/watch?v=5b8nL1xdZzI",
    "Agile": "https://www.youtube.com/watch?v=Z9QbYZh1YXY",
    "Communication": "https://www.youtube.com/watch?v=JOKGU3Hy7_8",
    "Writing": "https://www.youtube.com/watch?v=ksPhvD2r9OM",
    "Markdown": "https://www.youtube.com/watch?v=HUBNt18RFbo",
    "APIs": "https://www.youtube.com/watch?v=GZvSYJDk-us",
    "Technical Research": "https://www.youtube.com/watch?v=_GkxCIW46to",
    "Documentation": "https://www.youtube.com/watch?v=G3e-cpL7ofc",
    "Environmental Science": "https://www.youtube.com/watch?v=qA882Vj5lXk",
    "Policy": "https://www.youtube.com/watch?v=n3U4vHFTneA",
    "Carbon Analysis": "https://www.youtube.com/watch?v=yTZ5vwk5XqU",
    "Excel": "https://www.youtube.com/watch?v=hrzjtr1v3cE",
    "Accounting": "https://www.youtube.com/watch?v=4n2_2yuGvMU",
    "Financial Modeling": "https://www.youtube.com/watch?v=5W5hn63bErQ",
    "Valuation": "https://www.youtube.com/watch?v=JKiHZux8FvI",
    "Administration": "https://www.youtube.com/watch?v=9GzGAxF_3wM",
    "Health Policy": "https://www.youtube.com/watch?v=9yZ_bm7aU-I",
    "Leadership": "https://www.youtube.com/watch?v=ReRcHdeUG9Y",
    "Team Management": "https://www.youtube.com/watch?v=VyzK-SNqhXE",
    "Instructional Design": "https://www.youtube.com/watch?v=Cg7jM8DxEXM",
    "eLearning": "https://www.youtube.com/watch?v=HwFHZ4VfOro",
    "Curriculum Development": "https://www.youtube.com/watch?v=En3wXImZL1E",
    "Storyboarding": "https://www.youtube.com/watch?v=kuDxdjZ5JbE"
}

top_companies = {
    "Data Scientist": ["Google", "Microsoft", "Fractal Analytics"],
    "AI Engineer": ["OpenAI", "NVIDIA", "Meta"],
    "Cybersecurity Analyst": ["Palo Alto Networks", "Kaspersky", "TCS"],
    "Full Stack Developer": ["Amazon", "Zoho", "Infosys"],
    "Cloud Architect": ["AWS", "Azure", "Google Cloud"],
    "UI/UX Designer": ["Adobe", "Figma", "Airbnb"],
    "Robotics Engineer": ["Boston Dynamics", "ABB", "iRobot"],
    "Bioinformatics Specialist": ["Illumina", "Thermo Fisher", "Genentech"],
    "Digital Marketing Strategist": ["HubSpot", "Neil Patel Digital", "Moz"],
    "Product Manager": ["Google", "Flipkart", "Paytm"],
    "Technical Writer": ["Red Hat", "GitLab", "Microsoft"],
    "Sustainability Consultant": ["McKinsey", "EY", "BCG"],
    "Financial Analyst": ["Goldman Sachs", "JP Morgan", "Deloitte"],
    "Healthcare Operations Manager": ["Apollo Hospitals", "Fortis", "UnitedHealth"],
    "Learning Experience Designer": ["Coursera", "Khan Academy", "Duolingo"]
}
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/quiz")
def quiz():
    skills_list = list(mlb_skills.classes_)
    personality_list = list(mlb_personality.classes_)
    personality_meanings = {
        "E": "Extroversion",
        "I": "Introversion",
        "S": "Sensing",
        "N": "Intuition",
        "T": "Thinking",
        "F": "Feeling",
        "J": "Judging",
        "P": "Perceiving"
    }
    return render_template("index.html", skills=skills_list, personalities=personality_list,personality_meanings=personality_meanings)


@app.route("/predict", methods=["POST"])
def predict():
    skills = request.form.getlist("skills")
    personalities = request.form.getlist("personalities")
    expected_salary = int(request.form.get("expected_salary", 0))

    x_skills = mlb_skills.transform([skills])
    x_personality = mlb_personality.transform([personalities])
    x = np.hstack((x_skills, x_personality))

    probs = model.predict_proba(x)[0]
    top_indices = np.argsort(probs)[::-1][:2]

    best_career = le.inverse_transform([top_indices[0]])[0]
    alt_career = le.inverse_transform([top_indices[1]])[0]

    def get_missing_skills(career):
        req_skills = set(career_profiles.get(career, {}).get("skills", []))
        return list(req_skills - set(skills))

    def yt_links(skill_list):
        return {skill: youtube_links.get(skill, "") for skill in skill_list}

    result = {
        "career_match": best_career,
        "alt_career": alt_career,
        "confidence_best": f"{round(probs[top_indices[0]] * 100)}%",
        "confidence_alt": f"{round(probs[top_indices[1]] * 100)}%",
        "missing_skills_best": get_missing_skills(best_career),
        "missing_skills_alt": get_missing_skills(alt_career),
        "youtube_links_best": yt_links(get_missing_skills(best_career)),
        "youtube_links_alt": yt_links(get_missing_skills(alt_career)),
        "top_companies_best": top_companies.get(best_career, []),
        "top_companies_alt": top_companies.get(alt_career, [])
    }

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
