import React from 'react';
import Layout from '@theme/Layout';
import Head from '@docusaurus/Head';

export default function FAQs() {
  return (
    <Layout
      title="FAQs | VedicSkill Academy"
      description="Find answers to common questions about VedicSkill Academy courses, enrollment, certifications, and support."
    >
      <Head>
        <meta name="author" content="VedicSkill Academy" />
        <meta property="og:title" content="VedicSkill Academy FAQs" />
        <meta
          property="og:description"
          content="Get answers to frequently asked questions about VedicSkill Academy's AI, data science, and software development training."
        />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://courses.vedicskill.com/faqs" />
        <meta name="twitter:card" content="summary_large_image" />
      </Head>
      <main style={{ padding: '2rem', maxWidth: '820px', margin: '0 auto' }}>
        <h1>Frequently Asked Questions</h1>
        <p>
          Welcome to the VedicSkill Academy FAQs. Here you’ll find answers about our AI, data science, Python, MongoDB, and analytics courses, enrollment process, and support.
        </p>

        <ol>
          <li>
            <strong>What training does VedicSkill Academy offer?</strong><br />
            We offer premium tutorials and course guides for AI, machine learning, deep learning, generative AI, computer vision, Python programming, MongoDB, LangChain, and cloud deployment.
          </li>
          <li>
            <strong>Who is this training for?</strong><br />
            Our content is designed for learners, developers, data scientists, AI engineers, students, and working professionals who want industry-ready skills.
          </li>
          <li>
            <strong>How can I enroll in a course?</strong><br />
            Visit our <a href="https://www.udemy.com/user/freeai-space/" target="_blank" rel="noreferrer">Udemy profile</a>, choose your course, and enroll instantly.
          </li>
          <li>
            <strong>Do you provide certification?</strong><br />
            Courses purchased through Udemy come with Udemy’s certification and completion credentials where available.
          </li>
          <li>
            <strong>Is VedicSkill Academy suitable for beginners?</strong><br />
            Yes. We offer beginner-friendly introductions to Python, statistics, and AI, while also providing advanced modules for professionals.
          </li>
          <li>
            <strong>What support is available?</strong><br />
            We provide email support and documentation guidance for enrolled learners, and you can reach us at <a href="mailto:support@vedicskill.com">support@vedicskill.com</a>.
          </li>
          <li>
            <strong>Can I learn on mobile?</strong><br />
            Yes. Udemy mobile app access is supported for all courses, so you can learn anywhere.
          </li>
          <li>
            <strong>How often do you update content?</strong><br />
            We continuously refresh course material to keep pace with the latest AI, data science, and software development trends.
          </li>
          <li>
            <strong>Can I access notes and source code?</strong><br />
            Yes. Most courses provide downloadable notes, code examples, and project files, often accessible through GitHub and course resources.
          </li>
          <li>
            <strong>What is your refund policy?</strong><br />
            Udemy provides a 30-day refund policy for eligible course purchases.
          </li>
          <li>
            <strong>Do you offer enterprise or custom training?</strong><br />
            At the moment, our focus is on self-paced online tutorials and Udemy courses. Custom training requests can be shared through our contact email.
          </li>
          <li>
            <strong>How do I stay up to date with new courses?</strong><br />
            Subscribe to our YouTube channel at <a href="https://www.youtube.com/@datascienceanywhere" target="_blank" rel="noreferrer">Data Science Anywhere</a> and follow our Udemy profile.
          </li>
        </ol>

        <p>
          ✍️ Have another question? Contact us at <a href="mailto:support@vedicskill.com">support@vedicskill.com</a> and we’ll respond quickly.
        </p>
      </main>
    </Layout>
  );
}
