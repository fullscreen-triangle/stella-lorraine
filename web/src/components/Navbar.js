import Link from "next/link";
import { useRouter } from "next/router";
import { motion } from "framer-motion";
import { useThemeSwitch } from "./Hooks/useThemeSwitch";

const NavLink = ({ href, label }) => {
  const router = useRouter();
  const active = router.asPath === href;
  return (
    <Link
      href={href}
      className="relative group text-xs tracking-widest uppercase font-medium
                 text-dark dark:text-light opacity-50 hover:opacity-100
                 transition-opacity duration-200"
    >
      {label}
      <span
        className={`absolute left-0 -bottom-0.5 h-px bg-dark dark:bg-light
                    transition-[width] duration-300 ease
                    ${active ? "w-full" : "w-0 group-hover:w-full"}`}
      />
    </Link>
  );
};

export default function Navbar() {
  const [mode, setMode] = useThemeSwitch();

  return (
    <header
      className="w-full flex items-center justify-between
                 px-10 py-5 font-mont z-10
                 dark:text-light text-dark"
    >
      {/* Left: wordmark */}
      <Link
        href="/"
        className="text-xs tracking-widest uppercase opacity-30 hover:opacity-70
                   transition-opacity font-medium"
      >
        Categorical Spectrometry
      </Link>

      {/* Centre: instrument links */}
      <nav className="flex items-center gap-10">
        <NavLink href="/polymorphism" label="Polymorphism" />
        <NavLink href="/thin-film" label="Thin Film" />
        <NavLink href="/bioreactor" label="Bioreactor" />
        <NavLink href="/ritonavir" label="Synthesis" />
      </nav>

      {/* Right: theme toggle */}
      <motion.button
        onClick={() => setMode(mode === "light" ? "dark" : "light")}
        whileTap={{ scale: 0.9 }}
        className="w-5 h-5 rounded-full border border-current opacity-30
                   hover:opacity-70 transition-opacity flex items-center justify-center"
        aria-label="Toggle theme"
      >
        <span
          className="block w-2 h-2 rounded-full"
          style={{
            background: mode === "light" ? "#1b1b1b" : "#f5f5f5",
          }}
        />
      </motion.button>
    </header>
  );
}
