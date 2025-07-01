export default function DefaultLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="relative flex flex-col h-screen">
      {/*<Navbar />*/}
      <main className="container mx-auto max-w-7xl px-6 flex-grow py-6 flex-shrink h-0">
        {children}
      </main>
    </div>
  );
}
