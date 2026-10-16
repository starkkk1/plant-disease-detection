import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import "./globals.css";

const manrope = localFont({
  src: "./fonts/Manrope-Variable.ttf",
  variable: "--font-manrope",
  weight: "200 800",
  display: "swap",
});
const instrumentSerif = localFont({
  src: "./fonts/InstrumentSerif-Regular.ttf",
  variable: "--font-instrument-serif",
  weight: "400",
  display: "swap",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Leaf | Tomato leaf analysis",
  description: "Explore tomato leaf disease predictions from a photo.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0b100e", colorScheme: "dark" };

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${manrope.variable} ${instrumentSerif.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
