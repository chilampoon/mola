import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="italic font-semibold tracking-tight text-neutral-950 dark:text-neutral-50">
          Mola mola
        </span>
      )
    },
    themeSwitch: {
      enabled: true
    }
  };
}
