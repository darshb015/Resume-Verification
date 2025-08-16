// Simple mobile menu toggling + sticky header shadow on scroll
(function () {
  const hamburger = document.getElementById('hamburger');
  const mobileMenu = document.getElementById('mobile-menu');
  const header = document.getElementById('site-header');

  // Toggle mobile menu
  function toggleMenu() {
    const expanded = hamburger.getAttribute('aria-expanded') === 'true';
    hamburger.setAttribute('aria-expanded', String(!expanded));
    if (expanded) {
      mobileMenu.hidden = true;
      mobileMenu.setAttribute('aria-hidden', 'true');
      hamburger.setAttribute('aria-label', 'Open menu');
    } else {
      mobileMenu.hidden = false;
      mobileMenu.setAttribute('aria-hidden', 'false');
      hamburger.setAttribute('aria-label', 'Close menu');
      // move focus to first link for keyboard users
      const firstLink = mobileMenu.querySelector('a');
      if (firstLink) firstLink.focus();
    }
  }

  hamburger.addEventListener('click', toggleMenu);

  // Close menu on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (hamburger.getAttribute('aria-expanded') === 'true') {
        toggleMenu();
        hamburger.focus();
      }
    }
  });

  // Add subtle shadow to header when page is scrolled
  function onScroll() {
    if (window.scrollY > 6) header.classList.add('scrolled');
    else header.classList.remove('scrolled');
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
