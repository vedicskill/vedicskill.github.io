import React, { useMemo, useState } from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./index.module.css";

const config = require("../../docusaurus.config");

/*
 * Extract docs plugins automatically
 */
const courses = config.plugins
  .filter(
    (plugin) =>
      Array.isArray(plugin) &&
      plugin[0] === "@docusaurus/plugin-content-docs"
  )
  .map((plugin) => {
    const p = plugin[1];

    return {
      id: p.id,
      title:
        p.customFields?.title ||
        p.id
          .replace(/-/g, " ")
          .replace(/\b\w/g, (c) => c.toUpperCase()),

      description:
        p.customFields?.description ||
        "Professional project-based documentation and training.",

      route: `/${p.routeBasePath}`,
      icon: p.customFields?.icon || "📘",
      category: p.customFields?.category || "Course",
    };
  });

function CourseCard({ course }) {
  return (
    <Link to={course.route} className={styles.courseCardLink}>
      <div className={styles.courseCard}>
        <div className={styles.courseCardIcon}>{course.icon}</div>

        <h3 className={styles.courseCardTitle}>{course.title}</h3>

        <p className={styles.courseCardDescription}>{course.description}</p>

        <span className={styles.courseCardCategory}>{course.category}</span>
      </div>
    </Link>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  const [search, setSearch] = useState("");

  const filteredCourses = useMemo(() => {
    return courses.filter(
      (course) =>
        course.title.toLowerCase().includes(search.toLowerCase()) ||
        course.description.toLowerCase().includes(search.toLowerCase())
    );
  }, [search]);

  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}
    >
      <main>

        {/* Hero */}

        <section
          style={{
            paddingTop: "80px",
            paddingBottom: "60px",
            background: "var(--ifm-background-color)",
            borderBottom: "1px solid var(--ifm-color-emphasis-200)",
          }}
        >
          <div className="container">
            <div
              style={{
                maxWidth: "900px",
                margin: "0 auto",
                textAlign: "center",
              }}
            >
              <p
                style={{
                  marginBottom: "16px",
                  fontSize: "1rem",
                  fontWeight: 600,
                  color: "var(--ifm-color-primary)",
                }}
              >
                AI training · Data science tutorials · Python programming · MongoDB · Generative AI · Cloud deployment
              </p>
              <h1
                style={{
                  fontSize: "4rem",
                  fontWeight: 700,
                  marginBottom: "24px",
                  color: "var(--ifm-color-emphasis-900)",
                }}
              >
                VedicSkill Academy
              </h1>

              <p
                style={{
                  fontSize: "1.35rem",
                  lineHeight: 1.8,
                  color: "var(--ifm-color-emphasis-600)",
                  marginBottom: "32px",
                }}
              >
                Learn premium AI, machine learning, data science, computer vision, and analytics skills with structured tutorials, real-world projects, and deployment-ready training.
              </p>

              <div
                style={{
                  width: "100%",
                  maxWidth: "720px",
                  margin: "0 auto 32px",
                }}
              >
                <input
                  type="text"
                  placeholder="Search courses..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "18px 24px",
                    borderRadius: "14px",
                    border: "1px solid var(--ifm-color-emphasis-200)",
                    fontSize: "1rem",
                    outline: "none",
                    backgroundColor: "var(--ifm-background-color)",
                    color: "var(--ifm-color-emphasis-900)",
                  }}
                />
              </div>

              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "16px",
                  alignItems: "center",
                }}
              >
                <a
                  href="https://www.udemy.com/user/freeai-space/"
                  className="button button--primary button--lg"
                  style={{ minWidth: "240px" }}
                >
                  Start Learning on Udemy
                </a>
                <span
                  style={{
                    color: "var(--ifm-color-emphasis-600)",
                    fontSize: "0.95rem",
                  }}
                >
                  Explore career-focused AI, data science, and software development courses.
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* Courses */}

        <section
          style={{
            paddingTop: "60px",
            paddingBottom: "80px",
            background: "var(--ifm-background-color)",
          }}
        >
          <div className="container">

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "30px",
              }}
            >
              <h2
                style={{
                  margin: 0,
                  color: "var(--ifm-color-emphasis-900)",
                }}
              >
                Courses
              </h2>

              <span
                style={{
                  color: "var(--ifm-color-emphasis-600)",
                }}
              >
                {filteredCourses.length} Available
              </span>
            </div>

            <div className="row">
              {filteredCourses.map((course) => (
                <div
                  key={course.id}
                  className="col col--4 margin-bottom--lg"
                >
                  <CourseCard course={course} />
                </div>
              ))}
            </div>

          </div>
        </section>

      </main>
    </Layout>
  );
}