import dynamic from "next/dynamic";
import Head from "next/head";
import Link from "next/link";

const EscapementViewer = dynamic(
  () => import("@/components/EscapementViewer"),
  { ssr: false }
);

export default function Home() {
  return (
    <>
      <Head>
        <title>Categorical Spectrometry</title>
        <meta
          name="description"
          content="Physical measurement instruments grounded in bounded phase space geometry."
        />
      </Head>

      <div style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden" }}>
        <EscapementViewer />

        {/* Minimal nav anchored to bottom center */}
        <nav
          style={{
            position: "absolute",
            bottom: "2.5rem",
            left: "50%",
            transform: "translateX(-50%)",
            display: "flex",
            gap: "4rem",
            zIndex: 20,
          }}
        >
          <Link
            href="/polymorphism"
            style={{
              color: "#f5f5f5",
              fontSize: "0.68rem",
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              opacity: 0.55,
              textDecoration: "none",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = 0.55)}
          >
            Polymorphism
          </Link>
          <Link
            href="/thin-film"
            style={{
              color: "#f5f5f5",
              fontSize: "0.68rem",
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              opacity: 0.55,
              textDecoration: "none",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = 0.55)}
          >
            Thin Film
          </Link>
          <Link
            href="/bioreactor"
            style={{
              color: "#f5f5f5",
              fontSize: "0.68rem",
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              opacity: 0.55,
              textDecoration: "none",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = 0.55)}
          >
            Bioreactor
          </Link>
          <Link
            href="/ritonavir"
            style={{
              color: "#f5f5f5",
              fontSize: "0.68rem",
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              opacity: 0.55,
              textDecoration: "none",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = 0.55)}
          >
            Synthesis
          </Link>
        </nav>
      </div>
    </>
  );
}

// Bypass Navbar/Footer — landing page is the GLB viewer only
Home.getLayout = (page) => page;
