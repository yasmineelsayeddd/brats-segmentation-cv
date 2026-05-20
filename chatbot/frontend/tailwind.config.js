/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#f8f9fa',
          secondary: '#ffffff',
          tertiary: '#f1f3f5',
        },
        surface: {
          DEFAULT: '#ffffff',
          hover: '#f1f3f5',
          border: '#e9ecef',
        },
        accent: {
          DEFAULT: '#6366f1',
          hover: '#5558e6',
          muted: '#eef2ff',
        },
        text: {
          primary: '#1e293b',
          secondary: '#64748b',
          muted: '#94a3b8',
        },
        success: '#22c55e',
        warning: '#eab308',
        danger: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'soft': '0 1px 3px 0 rgb(0 0 0 / 0.03), 0 1px 2px -1px rgb(0 0 0 / 0.03)',
        'card': '0 1px 4px 0 rgb(0 0 0 / 0.04), 0 1px 3px -1px rgb(0 0 0 / 0.03)',
        'elevated': '0 4px 12px 0 rgb(0 0 0 / 0.05), 0 1px 3px 0 rgb(0 0 0 / 0.03)',
      },
    },
  },
  plugins: [],
}
