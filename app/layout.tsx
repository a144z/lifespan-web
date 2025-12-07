import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Lifespan Prediction - AI Face Analysis',
  description: 'Predict remaining lifespan from face images using lightweight AI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

