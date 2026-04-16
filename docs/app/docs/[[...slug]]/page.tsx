import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from "fumadocs-ui/page";

import { getDocPage, docsPages } from "@/lib/docs";
import { getMDXComponents } from "@/mdx-components";

export function generateStaticParams() {
  return docsPages.map((page) => ({ slug: page.slug }));
}

export async function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>;
}): Promise<Metadata> {
  const params = await props.params;
  const page = getDocPage(params.slug);

  if (!page) return {};

  return {
    title: page.title,
    description: page.description
  };
}

export default async function DocsSlugPage(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const params = await props.params;
  const page = getDocPage(params.slug);

  if (!page) notFound();

  const MDX = page.component;

  return (
    <DocsPage>
      <DocsTitle>{page.title}</DocsTitle>
      <DocsDescription>{page.description}</DocsDescription>
      <DocsBody>
        <MDX components={getMDXComponents()} />
      </DocsBody>
    </DocsPage>
  );
}
