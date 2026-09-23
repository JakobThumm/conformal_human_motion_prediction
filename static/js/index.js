// Smooth scrolling for in-page anchor links.
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
});

// ---------------------------------------------------------------------------
// Inline scenes cut from the supplementary film.
//
// There are ten of them on this page. Autoplaying all ten on load would pull
// ~2.5 MB and keep ten decoders busy, so each <video class="scene"> carries its
// URL in data-src and is only attached, and only played, while it is on screen.
// Everything degrades gracefully: without IntersectionObserver, or with reduced
// motion requested, the clips are loaded with controls and left paused.
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
  var scenes = Array.prototype.slice.call(document.querySelectorAll('video.scene'));
  if (!scenes.length) return;

  var reduceMotion = window.matchMedia &&
                     window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function attach(video) {
    if (video.dataset.src && !video.src) {
      video.src = video.dataset.src;
    }
  }

  if (reduceMotion || !('IntersectionObserver' in window)) {
    scenes.forEach(function (video) {
      attach(video);
      video.controls = true;      // let the reader start it themselves
      video.preload = 'metadata';
    });
    return;
  }

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      var video = entry.target;
      if (entry.isIntersecting) {
        attach(video);
        var p = video.play();
        // Autoplay can still be refused (e.g. data-saver); fall back to controls.
        if (p && typeof p.catch === 'function') {
          p.catch(function () { video.controls = true; });
        }
      } else if (!video.paused) {
        video.pause();
      }
    });
  }, { rootMargin: '150px 0px', threshold: 0.25 });

  scenes.forEach(function (video) { observer.observe(video); });
});
