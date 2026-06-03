import React, { useMemo, useState } from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

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
    <Link
      to={course.route}
      style={{
        textDecoration: "none",
        color: "inherit",
      }}
    >
      <div
        style={{
          height: "100%",
          border: "1px solid #E5E7EB",
          borderRadius: "18px",
          padding: "24px",
          background: "#FFFFFF",
          transition: "all 0.2s ease",
          cursor: "pointer",
          boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
        }}
      >
        <div
          style={{
            fontSize: "2rem",
            marginBottom: "16px",
          }}
        >
          {course.icon}
        </div>

        <h3
          style={{
            marginBottom: "12px",
            fontWeight: 600,
          }}
        >
          {course.title}
        </h3>

        <p
          style={{
            color: "#6B7280",
            minHeight: "60px",
            marginBottom: "20px",
          }}
        >
          {course.description}
        </p>

        <span
          style={{
            display: "inline-block",
            padding: "6px 12px",
            borderRadius: "999px",
            background: "#ECFDF5",
            color: "#065F46",
            fontSize: "0.8rem",
            fontWeight: 600,
          }}
        >
          {course.category}
        </span>
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
            background: "#FFFFFF",
            borderBottom: "1px solid #F3F4F6",
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
              <h1
                style={{
                  fontSize: "4rem",
                  fontWeight: 700,
                  marginBottom: "20px",
                }}
              >
                VedicSkill Academy
              </h1>

              <p
                style={{
                  fontSize: "1.3rem",
                  color: "#6B7280",
                  marginBottom: "40px",
                }}
              >
                Learn AI, Data Science, Statistics, Python,
                Computer Vision and Engineering through
                project-based documentation.
              </p>

              <input
                type="text"
                placeholder="Search courses..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{
                  width: "100%",
                  maxWidth: "700px",
                  padding: "18px 24px",
                  borderRadius: "14px",
                  border: "1px solid #D1D5DB",
                  fontSize: "1rem",
                  outline: "none",
                }}
              />
            </div>

          </div>
        </section>

        {/* Courses */}

        <section
          style={{
            paddingTop: "60px",
            paddingBottom: "80px",
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
                }}
              >
                Courses
              </h2>

              <span
                style={{
                  color: "#6B7280",
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