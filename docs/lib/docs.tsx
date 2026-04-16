import type { ComponentType } from "react";
import type { Root } from "fumadocs-core/page-tree";
import type { MDXComponents } from "mdx/types";

import ChangelogPage from "@/content/docs/changelog.mdx";
import CliIndexPage from "@/content/docs/cli/index.mdx";
import InferPhayesPage from "@/content/docs/cli/infer-phayes.mdx";
import InferPosteriorsPage from "@/content/docs/cli/infer-posteriors.mdx";
import InferSomaPage from "@/content/docs/cli/infer-soma.mdx";
import MutMapPage from "@/content/docs/cli/mut-map.mdx";
import MutWritePage from "@/content/docs/cli/mut-write.mdx";
import ParseCountPage from "@/content/docs/cli/parse-count.mdx";
import ReadAnnotatePage from "@/content/docs/cli/read-annotate.mdx";
import ReadDemuxPage from "@/content/docs/cli/read-demux.mdx";
import ReadSubsetPage from "@/content/docs/cli/read-subset.mdx";
import ReadTrimPage from "@/content/docs/cli/read-trim.mdx";
import InstallationPage from "@/content/docs/getting-started/installation.mdx";
import QuickstartPage from "@/content/docs/getting-started/quickstart.mdx";
import LongReadWorkflowPage from "@/content/docs/guides/long-read-workflow.mdx";
import ShortReadWorkflowPage from "@/content/docs/guides/short-read-workflow.mdx";
import DocsHomePage from "@/content/docs/index.mdx";

type MdxPageComponent = ComponentType<{ components?: MDXComponents }>;

export type DocPage = {
  description: string;
  group: "Getting started" | "Commands" | "Guides" | "Project";
  path: string;
  slug: string[];
  title: string;
  component: MdxPageComponent;
};

function docPage(
  path: string,
  title: string,
  description: string,
  group: DocPage["group"],
  component: MdxPageComponent
): DocPage {
  return {
    path,
    title,
    description,
    group,
    component,
    slug: path === "" ? [] : path.split("/")
  };
}

export const docsPages: DocPage[] = [
  docPage("", "mola", "Multi-modal Observations from Long- and short-read Alignments", "Project", DocsHomePage),
  docPage("getting-started/installation", "Installation", "Install mola from source and set up the Fumadocs app locally.", "Getting started", InstallationPage),
  docPage("getting-started/quickstart", "Quickstart", "End-to-end command shapes for the current short-read and long-read mola workflows.", "Getting started", QuickstartPage),
  docPage("cli", "Commands overview", "Command groups and per-command reference pages for the mola CLI.", "Commands", CliIndexPage),
  docPage("cli/read-demux", "mola read demux", "Demultiplex a BAM into separate files using an alignment tag.", "Commands", ReadDemuxPage),
  docPage("cli/read-subset", "mola read subset", "Extract a subset of alignments from a BAM using a read id list.", "Commands", ReadSubsetPage),
  docPage("cli/read-trim", "mola read trim", "Trim adapter-like sequence from BAM records and write trimmed FASTQ.", "Commands", ReadTrimPage),
  docPage("cli/read-annotate", "mola read annotate", "Build per-read objects and annotate reads against genes, repeats, and splice structure.", "Commands", ReadAnnotatePage),
  docPage("cli/mut-map", "mola mut map", "Map mismatch sites from a VCF back onto annotated read objects.", "Commands", MutMapPage),
  docPage("cli/mut-write", "mola mut write", "Export mismatch site summaries from the site object directory.", "Commands", MutWritePage),
  docPage("cli/infer-posteriors", "mola infer posteriors", "Fit beta-binomial mixture models and classify candidate sites.", "Commands", InferPosteriorsPage),
  docPage("cli/infer-phayes", "mola infer phayes", "Phase putative germline SNPs across long reads.", "Commands", InferPhayesPage),
  docPage("cli/infer-soma", "mola infer soma", "Test phased candidate sites for somatic-like behavior.", "Commands", InferSomaPage),
  docPage("cli/parse-count", "mola parse count", "Export bulk or single-cell count tables from annotated read objects.", "Commands", ParseCountPage),
  docPage("guides/short-read-workflow", "Short-read workflow", "Recommended command sequence for bulk and single-cell short-read data.", "Guides", ShortReadWorkflowPage),
  docPage("guides/long-read-workflow", "Long-read workflow", "Recommended command sequence for long-read runs that need posterior modeling and phasing.", "Guides", LongReadWorkflowPage),
  docPage("changelog", "Changelog", "Recent changes to the public mola repository.", "Project", ChangelogPage)
];

export const docsPageTree: Root = {
  name: "mola docs",
  children: [
    { type: "page", name: "Overview", url: "/docs" },
    {
      type: "folder",
      name: "Getting started",
      children: [
        { type: "page", name: "Installation", url: "/docs/getting-started/installation" },
        { type: "page", name: "Quickstart", url: "/docs/getting-started/quickstart" }
      ]
    },
    {
      type: "folder",
      name: "Guides",
      children: [
        { type: "page", name: "Short-read workflow", url: "/docs/guides/short-read-workflow" },
        { type: "page", name: "Long-read workflow", url: "/docs/guides/long-read-workflow" }
      ]
    },
    {
      type: "folder",
      name: "Commands",
      defaultOpen: false,
      children: [
        { type: "page", name: "Commands overview", url: "/docs/cli" },
        { type: "page", name: "mola read demux", url: "/docs/cli/read-demux" },
        { type: "page", name: "mola read subset", url: "/docs/cli/read-subset" },
        { type: "page", name: "mola read trim", url: "/docs/cli/read-trim" },
        { type: "page", name: "mola read annotate", url: "/docs/cli/read-annotate" },
        { type: "page", name: "mola mut map", url: "/docs/cli/mut-map" },
        { type: "page", name: "mola mut write", url: "/docs/cli/mut-write" },
        { type: "page", name: "mola infer posteriors", url: "/docs/cli/infer-posteriors" },
        { type: "page", name: "mola infer phayes", url: "/docs/cli/infer-phayes" },
        { type: "page", name: "mola infer soma", url: "/docs/cli/infer-soma" },
        { type: "page", name: "mola parse count", url: "/docs/cli/parse-count" }
      ]
    },
    { type: "page", name: "Changelog", url: "/docs/changelog" }
  ]
};

export function getDocPage(slug?: string[]): DocPage | undefined {
  const path = slug?.length ? slug.join("/") : "";
  return docsPages.find((page) => page.path === path);
}
