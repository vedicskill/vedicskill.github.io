// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

const {themes: prismThemes} = require('prism-react-renderer');

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Vedicskill',
  tagline: 'Vedicskill is practical way of learning skills',
  favicon: 'img/favicon_io/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://vedicskill.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/',
  trailingSlash: false, // Global trailingSlash config

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'vedicskill', // Usually your GitHub org/user name.
  projectName: 'vedicskill.github.io', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Presets: this is where classic preset + GA config goes
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: false, // we define custom docs plugins separately below
        // Blog configuration
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        // Sitemap configuration as per official docs
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          filename: 'sitemap.xml',
          ignorePatterns: ['/tags/**'], // Example: ignore tags pages
          lastmod: 'date', // 'date' = YYYY-MM-DD, 'datetime' = ISO 8601
        },
        // ✅ Google Analytics config
        gtag: {
          trackingID: 'G-CDYJXR6P96',
          anonymizeIP: true,
        },
      }),
    ],
  ],

  // Plugins: multiple docs plugins with IDs
  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'mongodb',
        path: 'docs-mongodb',
        routeBasePath: 'mongodb',
        sidebarPath: require.resolve('./sidebars.js'),
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'statistics',
        path: 'docs-statistics',
        routeBasePath: 'statistics',
        sidebarPath: require.resolve('./sidebars.js'),
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'microsoft-copilot-studio',
        path: 'docs-copilot-studio',
        routeBasePath: 'microsoft-copilot-studio',
        sidebarPath: require.resolve('./sidebars.js'),
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({

      metadata: [
        // Primary SEO tags
        {name: 'description', content: 'Learn Data Science, Statistics, AI, Web Apps with Vedicskill'},
        {name: 'keywords', content: 'Analysis, Statistics, Mathematics, Machine learning, Deep Learning, Generative AI, Hands-on courses, Python, Data Science Web Apps'},
        {name: 'author', content: 'VedicSkill'},
        {name: 'viewport', content: 'width=device-width, initial-scale=1.0'},

        // Open Graph / Facebook
        {property: 'og:type', content: 'website'},
        {property: 'og:locale', content: 'en_US'},
        {property: 'og:site_name', content: 'VedicSkill'},

        //        // Twitter Cards
        //        {name: 'twitter:card', content: 'summary_large_image'},
        //        {name: 'twitter:site', content: '@vedicskill'}, // Replace with your actual Twitter handle
        //        {name: 'twitter:creator', content: '@vedicskill'},

        // Additional SEO
//        {name: 'theme-color', content: '#FF6B35'},
        {name: 'apple-mobile-web-app-capable', content: 'yes'},
      ],
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Vedicskill',
        logo: {
          alt: 'vedicskill logo',
          src: 'img/logo1.png',
        },
        items: [
          {
            label: 'Docs',
            position: 'left',
            items: [
              { to: '/mongodb/intro', label: 'MongoDB' },
              { to: '/statistics/intro', label: 'Statistics' },
              { to: '/microsoft-copilot-studio/intro', label: 'Microsoft Copilot Studio' },
            ],
          },
          { to: '/blog', label: 'Blog', position: 'left' },
          { to: '/about', label: 'About', position: 'left' },
          {
            href: 'https://www.udemy.com/user/freeai-space/',
            label: 'Udemy Courses',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              { label: 'MongoDB', to: '/mongodb/intro' },
              { label: 'Statistics', to: '/statistics/intro' },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'YouTube',
                href: 'https://www.youtube.com/@datascienceanywhere',
              },
              {
                label: 'Udemy',
                href: 'https://www.udemy.com/user/freeai-space/',
              },
              {
                label: 'email',
                href: 'mailto:support@vedicskill.com',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/vedicskill',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Vedicskill | Datascience Anywhere.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],
};

module.exports = config;
