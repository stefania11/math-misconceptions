import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Math Misconception Detector',
  description: 'Interactive prototype for middle schoolers to learn algebra using multimodal AI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-100">{children}</body>
    </html>
  );
}
