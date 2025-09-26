import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import styles from './index.module.css';
import { useState } from 'react';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner, styles.ctaSection)}>
      <div className={styles.heroBackground}></div>
      <div className="container">
        <Heading
          as="h1"
          className={clsx("hero__title", styles.sectionTitle, styles.alwaysWhiteText)}
          style={{
            fontSize: '4rem',
            marginBottom: '2rem'
          }}
        >
          {siteConfig.title}
        </Heading>
        <p
          className={clsx("hero__subtitle", styles.sectionSubtitle, styles.alwaysWhiteText)}
          style={{
            fontSize: '1.5rem',
            maxWidth: '800px'
          }}
        >
          {siteConfig.tagline}
        </p>
        <div className={clsx(styles.buttons, "margin-top--lg")}>
          <Link
            className={clsx("button button--lg", styles.enhancedButton)}
            style={{
              backgroundColor: "var(--ifm-color-white)",
              color: "var(--ifm-color-primary)",
              marginRight: '1rem'
            }}
            to="/"
          >
            Get Started →
          </Link>
          <Link
            className={clsx("button button--outline button--lg", styles.enhancedButton)}
            to="https://www.udemy.com/user/freeai-space/"
          >
            View Udemy Courses
          </Link>
        </div>
      </div>
    </header>
  );
}

function FeatureCard({ title, description, highlight = false }) {
  return (
    <div className={clsx(styles.featureCard, highlight && styles.featureCardHighlight)}>
      <div className="card__body">
        <h3
          className={clsx("text--center", styles.sectionTitle)}
          style={{fontSize: '1.4rem', marginBottom: '1rem'}}
        >
          {title}
        </h3>
        <p className="text--center" style={{lineHeight: '1.6'}}>
          {description}
        </p>
      </div>
    </div>
  );
}

function StatCard({ number, label }) {
  return (
    <div className={styles.statCard}>
      <div className={styles.statNumber}>{number}</div>
      <div className={styles.statLabel}>{label}</div>
    </div>
  );
}

function TestimonialCard({ name, role, content, rating }) {
  return (
    <div className={styles.testimonialCard}>
      <div className={styles.testimonialStars}>
        {[...Array(rating)].map((_, i) => (
          <span key={i}>★</span>
        ))}
      </div>
      <p className={styles.testimonialQuote}>"{content}"</p>
      <div className={styles.testimonialName}>{name}</div>
      <div className={styles.testimonialRole}>{role}</div>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  const [open, setOpen] = useState(false);

  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Master real-world skills through project-based learning with VedicSkill"
    >

      <HomepageHeader />



{/* Floating Chat Button */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          style={{
            position: "fixed",
            bottom: "20px",
            right: "20px",
            padding: "14px 18px",
            borderRadius: "50%",
            background: "#0E2841",
            color: "#fff",
            border: "none",
            cursor: "pointer",
            fontSize: "20px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
            zIndex: 1000,
          }}
        >
          💬
        </button>
      )}


{/* Floating Chat Window */}
      {open && (
        <div
          style={{
            position: "fixed",
            bottom: "80px",
            right: "20px",
            width: "350px",
            height: "500px",
            background: "#fff",
            border: "1px solid #ccc",
            borderRadius: "12px",
            overflow: "hidden",
            boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
            zIndex: 1000,
          }}
        >
          {/* Close button */}
          <button
            onClick={() => setOpen(false)}
            style={{
              position: "absolute",
              top: "5px",
              right: "10px",
              background: "transparent",
              border: "none",
              fontSize: "18px",
              cursor: "pointer",
              color: "#0E2841",
              zIndex: 1001,
            }}
          >
            ✖
          </button>

          {/* Copilot iframe */}
          <iframe
            src="https://copilotstudio.microsoft.com/environments/6a9f1beb-3cb5-e242-8c09-536bfd8149de/bots/cr3ae_agent/webchat?__version__=2"
            frameBorder="0"
            style={{ width: "100%", height: "100%" }}
            title="Microsoft Copilot Studio Chat"
            allow="microphone; camera"
            loading="lazy"
          />
        </div>
      )}

      {/* Main Features Section */}

      <section className="hero hero--secondary" style={{padding: '5rem 0'}}>
        <div className="container">
          <div className="text--center margin-bottom--xl">
            <Heading as="h2" className={styles.sectionTitle}>
              Why Choose Project-Based Learning?
            </Heading>
            <p className={styles.sectionSubtitle}>
              In the real world, 70% of professionals struggle because of lack of practical exposure and resilience.
              Our courses contain projects that help you understand challenges and mitigation techniques.
            </p>
          </div>

          <div className="row">
            <div className="col col--4" >
              <FeatureCard
                title="Hands-on Projects"
                description="Build real-world applications and solve practical problems that mirror industry challenges."
                highlight={true}
              />
            </div>
            <div className="col col--4">
              <FeatureCard
                title="Industry Mentorship"
                description="Learn from experienced professionals who guide you through complex scenarios and best practices."
              />
            </div>
            <div className="col col--4">
              <FeatureCard
                title="Career Ready Skills"
                description="Develop the resilience and problem-solving abilities that make you comfortable with any project."
              />
            </div>
          </div>
        </div>
      </section>

      {/* Statistics Section */}
      <section className={styles.statsSection}>
        <div className="container">
          <div className="row">
            <div className="col col--3">
              <StatCard number="1000+" label="Students Trained" />
            </div>
            <div className="col col--3">
              <StatCard number="50+" label="Industry Projects" />
            </div>
            <div className="col col--3">
              <StatCard number="95%" label="Job Placement Rate" />
            </div>
            <div className="col col--3">
              <StatCard number="4.8" label="Average Rating" />
            </div>
          </div>
        </div>
      </section>

      {/* Learning Approach Section */}
      <section className={styles.approachSection}>
        <div className="container">
          <div className="row">
            <div className="col col--6">
              <Heading as="h2" className={clsx(styles.sectionTitle, "margin-bottom--lg")} style={{textAlign: 'left'}}>
                Our Learning Approach
              </Heading>
              <div className="margin-bottom--md">
                <div className={styles.approachItem}>
                  <span className={styles.checkIcon}>✅</span>
                  <div>
                    <h3 style={{marginBottom: '0.5rem'}}>Real-world Scenarios</h3>
                    <p style={{margin: 0}}>Work on projects that simulate actual industry challenges and requirements.</p>
                  </div>
                </div>
                <div className={styles.approachItem}>
                  <span className={styles.checkIcon}>✅</span>
                  <div>
                    <h3 style={{marginBottom: '0.5rem'}}>Practical Exposure</h3>
                    <p style={{margin: 0}}>Gain hands-on experience with tools and technologies used in professional environments.</p>
                  </div>
                </div>
                <div className={styles.approachItem}>
                  <span className={styles.checkIcon}>✅</span>
                  <div>
                    <h3 style={{marginBottom: '0.5rem'}}>Problem-solving Skills</h3>
                    <p style={{margin: 0}}>Develop resilience and critical thinking through challenging project scenarios.</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="col col--6">
              <div className={styles.videoCard}>
                <div className={styles.videoPlaceholder}>▶</div>
                <h3 style={{marginBottom: '1rem'}}>See Our Method in Action</h3>
                <p style={{marginBottom: '1.5rem'}}>Watch how our project-based approach transforms theoretical knowledge into practical skills.</p>
                <Link
                  to="https://www.youtube.com/@datascienceanywhere"
                  className={clsx("button button--primary", styles.enhancedButton)}
                >
                  Watch Demo →
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Features Section */}
      <section className="hero" style={{padding: '5rem 0'}}>
        <div className="container">
          <div className="text--center margin-bottom--xl">
            <Heading as="h2" className={styles.sectionTitle}>What Makes Us Different</Heading>
            <p className={styles.sectionSubtitle}>Discover the VedicSkill advantage in professional education</p>
          </div>
          <div className="row">
            <div className="col col--4">
              <div className="text--center padding--lg">
                <div className={clsx(styles.featureIcon, styles.featureIconPrimary)}>🎯</div>
                <h3 style={{marginBottom: '1rem'}}>Targeted Learning Paths</h3>
                <p>Customized curriculum designed to meet specific industry requirements and career goals.</p>
              </div>
            </div>
            <div className="col col--4">
              <div className="text--center padding--lg">
                <div className={clsx(styles.featureIcon, styles.featureIconSecondary)}>⚡</div>
                <h3 style={{marginBottom: '1rem'}}>Fast-Track Progress</h3>
                <p>Accelerated learning methodology that gets you job-ready in minimal time.</p>
              </div>
            </div>
            <div className="col col--4">
              <div className="text--center padding--lg">
                <div className={clsx(styles.featureIcon, styles.featureIconTertiary)}>🛡️</div>
                <h3 style={{marginBottom: '1rem'}}>Quality Assurance</h3>
                <p>Rigorous quality standards ensure you receive the best education and support.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="hero hero--secondary" style={{padding: '5rem 0'}}>
        <div className="container">
          <div className="text--center margin-bottom--xl">
            <Heading as="h2" className={styles.sectionTitle}>What Our Students Say</Heading>
            <p className={styles.sectionSubtitle}>Hear from professionals who transformed their careers with VedicSkill</p>
          </div>
          <div className="row">
            <div className="col col--4">
              <TestimonialCard
                name="Sarah Johnson"
                role="Software Developer at TechCorp"
                content="The project-based approach gave me the confidence to tackle complex problems in my new role. I felt prepared from day one."
                rating={5}
              />
            </div>
            <div className="col col--4">
              <TestimonialCard
                name="Mike Chen"
                role="Data Scientist at Analytics Pro"
                content="Unlike other courses, this actually prepared me for real challenges. The projects were incredibly valuable for my career growth."
                rating={5}
              />
            </div>
            <div className="col col--4">
              <TestimonialCard
                name="Emily Rodriguez"
                role="Full Stack Developer at StartupX"
                content="The hands-on experience and mentorship made all the difference. I landed my dream job within months of completing the course."
                rating={5}
              />
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className={styles.ctaSection}>
        <div className="container">
          <div className="text--center">
            <Heading as="h2" className="hero__title" style={{marginBottom: '1.5rem'}}>Ready to Start Your Journey?</Heading>
            <p className="hero__subtitle margin-bottom--lg" style={{maxWidth: '600px', margin: '0 auto 2rem'}}>
              Join thousands of students who have built successful careers through practical, project-based learning.
            </p>
            <div className={styles.buttons}>
              <Link
                className={clsx("button button--secondary button--lg", styles.enhancedButton)}
                style={{backgroundColor: 'var(--ifm-color-white)', color: 'var(--ifm-color-primary)', marginRight: '1rem'}}
                to="/"
              >
                Browse Courses
              </Link>
              <Link
                className={clsx("button button--outline button--secondary button--lg", styles.enhancedButton)}
                to="https://www.udemy.com/user/freeai-space/"
              >
                View Udemy Courses
              </Link>
            </div>
          </div>
        </div>
      </section>

      <HomepageFeatures />
    </Layout>
  );
}