import React from 'react';
import DocItemPre from '@docusaurus/theme-classic/lib/theme/DocItem/Layout';
import ProtectedContent from '../../../components/ProtectedContent';

/**
 * DocItem Component (Swizzled)
 * 
 * This component swizzles Docusaurus's default DocItem component to add
 * automatic protection based on the document's frontmatter.
 * 
 * When a document contains `requiresLogin: true` in its frontmatter,
 * the document will automatically be wrapped with the ProtectedContent component.
 * 
 * File location: src/theme/DocItem/Layout/index.tsx
 * 
 * How it works:
 * 1. Extracts requiresLogin from doc metadata
 * 2. If requiresLogin is true, wraps the doc with ProtectedContent
 * 3. If requiresLogin is false or undefined, renders normally
 */
export default function DocItemLayout(props: any): JSX.Element {
  // Extract the document metadata
  const { metadata } = props;
  
  // Get requiresLogin from frontmatter (default to false for backward compatibility)
  const requiresLogin = metadata?.frontMatter?.requiresLogin ?? false;

  // Wrap with ProtectedContent if requiresLogin is true
  return (
    <ProtectedContent requiresLogin={requiresLogin}>
      <DocItemPre {...props} />
    </ProtectedContent>
  );
}
