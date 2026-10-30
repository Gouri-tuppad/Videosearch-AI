export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      boxShadow: {
        glass: "0 25px 60px rgba(15, 23, 42, 0.35)",
      },
      backgroundImage: {
        "radial-fade": "radial-gradient(circle at top left, rgba(96, 165, 250, 0.14), transparent 28%), radial-gradient(circle at bottom right, rgba(168, 85, 247, 0.16), transparent 30%)",
      },
      colors: {
        surface: "rgba(15, 23, 42, 0.82)",
      },
    },
  },
  plugins: [],
};
