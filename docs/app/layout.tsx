import type { Metadata } from "next";
import type { ReactNode } from "react";
import { RootProvider } from "fumadocs-ui/provider/next";

import "./global.css";

export const metadata: Metadata = {
  title: {
    default: "mola docs",
    template: "%s | mola docs"
  },
  description:
    "Documentation for mola."
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className="min-h-screen bg-white text-neutral-950 antialiased dark:bg-neutral-950 dark:text-neutral-50"
      >
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
