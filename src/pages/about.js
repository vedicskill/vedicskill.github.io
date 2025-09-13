import React from 'react';
import Layout from '@theme/Layout';

export default function About() {
  return (
    <Layout title="About Us" description="Learn more about Vedicskill">
      <main style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
        <h1>About Us</h1>
        <p>
          Vedicskill is a global education initiative dedicated to empowering learners with the knowledge and skills required for the future of work.
        </p>

        <h2>What We Do</h2>
        <p>
          At Vedicskill, we design and deliver courses that help learners gain hands-on expertise in the most in-demand fields:
        </p>
        <ul>
          <li><strong>Data Science & Analytics</strong> – Learn to harness data for smarter decision-making.</li>
          <li><strong>Artificial Intelligence & Machine Learning</strong> – Build intelligent systems that solve real-world problems.</li>
          <li><strong>Statistics</strong> – Strengthen your foundation with the language of data.</li>
          <li><strong>Business Intelligence Dashboards</strong> – Visualize insights and drive business growth.</li>
          <li><strong>Generative AI Applications</strong> – Explore cutting-edge innovations shaping tomorrow’s world.</li>
        </ul>

        <h2>Why Vedicskill?</h2>
        <ul>
          <li>🌍 <strong>Global Reach:</strong> Trusted by thousands of learners across the world.</li>
          <li>🎓 <strong>20+ Udemy Courses:</strong> Covering beginner to advanced levels in multiple domains.</li>
          <li>⚡ <strong>Practical Learning:</strong> Project-based courses designed for real-world applications.</li>
          <li>🤝 <strong>Community & Support:</strong> A growing learner community with continuous guidance.</li>
        </ul>

        <h2>Our Commitment</h2>
        <p>
          We believe that education is the greatest equalizer. By bridging the gap between theory and practice, we prepare individuals and organizations to thrive in the evolving digital landscape.
        </p>

        <p>
          📌 Curious about our policies, certifications, and support? Check out our <a href="/faqs">FAQs</a>.<br/>
          🌐 Ready to start your journey? Explore our courses here 👉 <a href="https://www.udemy.com/user/freeai-space/" target="_blank" rel="noreferrer">Vedicskill on Udemy</a>
        </p>
      </main>
    </Layout>
  );
}
