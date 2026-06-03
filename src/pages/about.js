import React from 'react';
import Layout from '@theme/Layout';

export default function About() {
  return (
    <Layout
      title="About VedicSkill Academy | Tutorials"
      description="Learn how VedicSkill Academy helps learners master AI, data science, machine learning, Python, MongoDB, and cloud deployment."
    >
      <main style={{ padding: '2rem', maxWidth: '820px', margin: '0 auto' }}>
        <h1>About VedicSkill Academy | Tutorials</h1>
        <p>
          VedicSkill Academy is a premium education platform for learners, developers, data scientists, and AI professionals. We deliver modern AI, machine learning, data science, Python, MongoDB, and analytics training designed to accelerate your career.
        </p>

        <h2>Our Mission</h2>
        <p>
          We help aspiring professionals build real-world skills through structured tutorials, hands-on projects, and industry-aligned learning paths. Our goal is to make advanced technologies accessible and practical for students, working professionals, and developers.
        </p>

        <h2>What We Teach</h2>
        <ul>
          <li><strong>Artificial Intelligence & Machine Learning</strong> – Build intelligent systems with AI models, neural networks, and deep learning.</li>
          <li><strong>Generative AI & LLMs</strong> – Learn advanced prompt engineering, LangChain, and multimodal AI applications.</li>
          <li><strong>Data Science & Analytics</strong> – Analyze data, build dashboards, and create data-driven insights.</li>
          <li><strong>Computer Vision</strong> – Develop image recognition, detection, and visual AI systems.</li>
          <li><strong>Python Programming</strong> – Master Python for coding, automation, and AI application development.</li>
          <li><strong>MongoDB & Cloud Databases</strong> – Build scalable data architectures, vector search, and modern app backends.</li>
        </ul>

        <h2>Why Choose Us</h2>
        <ul>
          <li><strong>Project-Based Training:</strong> Learn by doing with practical exercises, real code, and deployable solutions.</li>
          <li><strong>Career-Focused Content:</strong> Develop skills that align with software engineering, AI engineering, and data science roles.</li>
          <li><strong>Modern Technology Stack:</strong> Python, MongoDB, LangChain, AWS, OpenAI, computer vision, and analytics tools.</li>
          <li><strong>Flexible Learning:</strong> Self-paced tutorials suitable for beginners and advanced learners alike.</li>
        </ul>

        <h2>Trusted Learning Experience</h2>
        <p>
          Our training is built for learners who want to move beyond theory and launch practical AI and data science projects. Whether you're building ML pipelines, deploying web apps, or mastering generative AI, VedicSkill Academy equips you with the tools to succeed.
        </p>

        <p>
          📌 Explore course details in our <a href="/course-visual-ai-with-transformers-llms">Visual AI course</a>, <a href="/python-courses">Python tutorials</a>, and <a href="/mongodb">MongoDB training</a>.<br />
          🌐 Ready to gain in-demand skills? Visit our <a href="https://www.udemy.com/user/freeai-space/" target="_blank" rel="noreferrer">Udemy course library</a> to enroll now.
        </p>
      </main>
    </Layout>
  );
}
