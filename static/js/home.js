document.addEventListener('DOMContentLoaded', function() {
    // Find the Start Driving button by its text and class
    const drivingBtn = Array.from(document.querySelectorAll('.action-button.primary'))
        .find(btn => btn.textContent.includes('Start Driving'));
    if (drivingBtn) {
        drivingBtn.addEventListener('click', function(event) {
            event.preventDefault();
            window.location.href = '/index';
        });
    }

    // Theme toggle logic
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const storedTheme = localStorage.getItem('theme');
        // Set initial theme
        if (storedTheme) {
            document.body.classList.toggle('dark-mode', storedTheme === 'dark');
            updateThemeIcon(storedTheme === 'dark');
        } else {
            document.body.classList.toggle('dark-mode', prefersDarkScheme);
            updateThemeIcon(prefersDarkScheme);
            localStorage.setItem('theme', prefersDarkScheme ? 'dark' : 'light');
        }
        themeToggle.addEventListener('click', function() {
            const isDarkMode = !document.body.classList.contains('dark-mode');
            document.body.classList.toggle('dark-mode', isDarkMode);
            updateThemeIcon(isDarkMode);
            localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
        });
        function updateThemeIcon(isDarkMode) {
            const icon = themeToggle.querySelector('i');
            if (icon) {
                icon.className = isDarkMode ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
    }

    // --- Animation: Section Entrance ---
    // Animate hero, sections, and footer on load
    const animatedSections = [
        document.querySelector('.hero-content'),
        document.getElementById('features'),
        document.getElementById('how-it-works'),
        document.getElementById('safety'),
        document.getElementById('about'),
        document.querySelector('.main-footer')
    ];
    animatedSections.forEach((el, i) => {
        if (el) {
            el.style.animationDelay = (0.1 + i * 0.2) + 's';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }
    });

    // --- Animation: Animate on Scroll ---
    // Add .animate-on-scroll to feature cards, steps, stat cards
    document.querySelectorAll('.feature-card, .step, .stat-card').forEach(el => {
        el.classList.add('animate-on-scroll');
    });
    // Intersection Observer for scroll-in animation
    const observer = new window.IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15 });
    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}); 