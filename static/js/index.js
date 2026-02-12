window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    var paperIFrame = document.getElementById('paper-iframe');
    var paperPlaceholder = document.getElementById('paper-placeholder');
    if (paperIFrame && paperPlaceholder) {
        var paperSrc = paperIFrame.getAttribute('src');
        if (paperSrc) {
            var canLoadPaper = function() {
                return fetch(paperSrc, { method: 'HEAD' })
                    .then(function(res) {
                        if (res && res.ok) return true;
                        // Some static hosts may not support HEAD; try a tiny GET request.
                        if (res && (res.status === 403 || res.status === 405)) {
                            return fetch(paperSrc, { method: 'GET', headers: { Range: 'bytes=0-0' } })
                                .then(function(r) { return !!(r && (r.ok || r.status === 206)); });
                        }
                        return false;
                    })
                    .catch(function() { return false; });
            };

            canLoadPaper().then(function(ok) {
                if (!ok) return;
                paperIFrame.style.display = 'block';
                paperPlaceholder.style.display = 'none';
            });
        }
    }

})
