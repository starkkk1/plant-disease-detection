"use client";

import { type HTMLAttributes, useEffect, useRef } from "react";

/** Progressive enhancement: content remains readable without animation support. */
export function ScrollReveal({ className = "", children, onFocusCapture, ...props }: HTMLAttributes<HTMLElement>) {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node || !window.IntersectionObserver || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        node.dataset.reveal = "visible";
        observer.disconnect();
      }
    }, { threshold: 0.08, rootMargin: "0px 0px -20px 0px" });
    node.dataset.reveal = "pending";
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return <section {...props} ref={ref} className={`leaf-reveal ${className}`} onFocusCapture={(event) => {
    // Keyboard navigation must never leave focused content visually hidden.
    if (event.target instanceof HTMLElement && event.target.matches(":focus-visible")) {
      event.currentTarget.dataset.reveal = "visible";
    }
    onFocusCapture?.(event);
  }}>{children}</section>;
}
