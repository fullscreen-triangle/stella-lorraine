import "@/styles/globals.css";
import { AnimatePresence } from "framer-motion";
import { Montserrat } from "next/font/google";
import Head from "next/head";
import { useRouter } from "next/router";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const montserrat = Montserrat({ subsets: ["latin"], variable: "--font-mont" });

const DefaultLayout = ({ children, routerKey }) => (
  <>
    <Navbar />
    <AnimatePresence initial={false} mode="wait">
      <div key={routerKey}>{children}</div>
    </AnimatePresence>
    <Footer />
  </>
);

export default function App({ Component, pageProps }) {
  const router = useRouter();

  const getLayout = Component.getLayout;

  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main
        className={`${montserrat.variable} font-mont bg-dark w-full min-h-screen`}
      >
        {getLayout ? (
          getLayout(<Component {...pageProps} />)
        ) : (
          <DefaultLayout routerKey={router.asPath}>
            <Component {...pageProps} />
          </DefaultLayout>
        )}
      </main>
    </>
  );
}
