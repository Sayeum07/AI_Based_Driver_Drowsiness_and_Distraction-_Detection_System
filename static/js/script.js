// Modern Dashboard UI Script for Driver Monitoring System
// Version 5.0 (Enhanced with Animations)

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const sidebarNav = document.querySelector('.sidebar-nav');
    const menuToggle = document.getElementById('menuToggle');
    const appContainer = document.querySelector('.app-container');
    const contentSections = document.querySelectorAll('.content-section');
    const pageTitle = document.querySelector('.page-title');
    const themeToggle = document.getElementById('themeToggle');

    // Camera elements - update for dual cameras
    const drowsinessOverlay = document.getElementById('drowsinessOverlay');
    const distractionOverlay = document.getElementById('distractionOverlay');
    const drowsinessFeed = document.getElementById('drowsinessFeed');
    const distractionFeed = document.getElementById('distractionFeed');

    const notificationArea = document.getElementById('notificationArea');
    const drowsinessBox = document.getElementById('drowsinessBox');
    const drowsinessLevel = document.getElementById('drowsinessLevel');
    const drowsinessDescription = document.getElementById('drowsinessDescription');
    const drowsinessProgress = document.getElementById('drowsinessProgress');
    const distractionBox = document.getElementById('distractionBox');
    const distractionLevel = document.getElementById('distractionLevel');
    const distractionDescription = document.getElementById('distractionDescription');
    const distractionProgress = document.getElementById('distractionProgress');
    const cameraStatus = document.getElementById('cameraStatus');
    const cameraStatusText = document.getElementById('cameraStatusText');
    const statusCards = document.querySelectorAll('.status-card');
    const welcomeBanner = document.querySelector('.welcome-banner');
    const startMonitoringBtn = document.getElementById('startMonitoring');
    const distractionAlert = document.getElementById('distractionAlert');
    const alertMessage = document.getElementById('alertMessage');
    const closeAlert = document.getElementById('closeAlert');
    const logo = document.querySelector('.sidebar .logo');
    
    // Add monitoring state variable at the top with other variables
    let isMonitoringActive = false;

    // Initialize the app
    initApp();

    function initApp() {
        // Add animation classes to elements with a stagger effect
        animateOnLoad();
        
        // Setup observer for scroll animations
        initScrollAnimations();
        
        // Set up navigation
        setupNavigation();
        
        // Initialize theme
        initTheme();
        
        // Initialize camera state (but don't start it yet)
        initCameraState();
        
        // Initialize camera controls for zoom functionality
        setupCameraControls();
        
        // Initialize interactive elements
        initInteractiveElements();

        if (logo) {
            logo.style.cursor = 'pointer';
            logo.addEventListener('click', function() {
                window.location.href = './home.html';
            });
        }
    }

    // Initial load animations with stagger effect
    function animateOnLoad() {
        const elementsToAnimate = [
            welcomeBanner,
            ...document.querySelectorAll('.camera-card, .status-card, .settings-card')
        ].filter(Boolean);
        
        elementsToAnimate.forEach((element, index) => {
            element.style.opacity = '0';
            element.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }, 100 * index); // Stagger effect
        });
    }
    
    // Setup scroll reveal animations
    function initScrollAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('reveal-animation');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        
        document.querySelectorAll('.camera-card, .status-card, .settings-card, .setting-item').forEach(item => {
            item.classList.add('reveal-element');
            observer.observe(item);
        });
    }

    // Interactive elements initialization
    function initInteractiveElements() {
        // Make cards slightly tilt on hover using mouse position
        document.querySelectorAll('.camera-card, .status-card, .settings-card').forEach(card => {
            card.addEventListener('mousemove', handleCardTilt);
            card.addEventListener('mouseleave', resetCardTilt);
        });
        
        // Make nav items more interactive
        document.querySelectorAll('.nav-item').forEach(navItem => {
            navItem.addEventListener('mouseenter', (e) => {
                const icon = e.currentTarget.querySelector('i');
                if (icon) {
                    icon.style.transform = 'scale(1.2) translateX(3px)';
                    icon.style.transition = 'transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
                }
            });
            
            navItem.addEventListener('mouseleave', (e) => {
                const icon = e.currentTarget.querySelector('i');
                if (icon) {
                    icon.style.transform = '';
                }
            });
        });
        
        // Start monitoring button
        if (startMonitoringBtn) {
            startMonitoringBtn.addEventListener('click', function() {
                // Toggle button text and icon
                const icon = this.querySelector('i');
                if (icon.classList.contains('fa-play-circle')) {
                    // Start monitoring
                    icon.className = 'fas fa-pause-circle';
                    this.innerHTML = `<i class="fas fa-pause-circle"></i> Pause Monitoring`;
                    isMonitoringActive = true;
                    activateCamera();
                } else {
                    // Stop monitoring
                    icon.className = 'fas fa-play-circle';
                    this.innerHTML = `<i class="fas fa-play-circle"></i> Start Monitoring`;
                    isMonitoringActive = false;
                    deactivateCamera();
                    
                    // Hide any active alerts
                    hideDistractionAlert();
                }
                
                // Add ripple effect
                addRippleEffect(this);
            });
        }
        
        // Close alert button
        if (closeAlert) {
            closeAlert.addEventListener('click', function() {
                hideDistractionAlert();
            });
        }
        
        // Attach action links to navigation
        document.querySelectorAll('.action-link').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const section = this.getAttribute('data-section');
                if (section) {
                    // Find the corresponding nav item and trigger a click
                    const navItem = document.querySelector(`.nav-item[data-section="${section}"]`);
                    if (navItem) {
                        navItem.click();
                    }
                }
            });
        });
    }
    
    // Handle card tilt effect based on mouse position
    function handleCardTilt(e) {
        const card = e.currentTarget;
        const cardRect = card.getBoundingClientRect();
        const cardCenterX = cardRect.left + cardRect.width / 2;
        const cardCenterY = cardRect.top + cardRect.height / 2;
        const mouseX = e.clientX - cardCenterX;
        const mouseY = e.clientY - cardCenterY;
        
        // Calculate tilt based on mouse position (max 5 degrees)
        const tiltX = (mouseY / (cardRect.height / 2)) * 3;
        const tiltY = -(mouseX / (cardRect.width / 2)) * 3;
        
        // Apply the tilt effect
        card.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateZ(10px)`;
        
        // Add a subtle highlight effect where the cursor is
        const glareX = (mouseX / cardRect.width) * 100 + 50;
        const glareY = (mouseY / cardRect.height) * 100 + 50;
        card.style.background = `radial-gradient(circle at ${glareX}% ${glareY}%, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 50%)`;
    }
    
    // Reset card tilt on mouse leave
    function resetCardTilt(e) {
        const card = e.currentTarget;
        card.style.transform = '';
        card.style.background = '';
    }

    // Navigation setup
    function setupNavigation() {
        // Handle sidebar navigation
        if (sidebarNav) {
            sidebarNav.querySelectorAll('.nav-item').forEach(item => {
                item.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    // Update active nav item
                    sidebarNav.querySelectorAll('.nav-item').forEach(navItem => {
                        navItem.classList.remove('active');
                    });
                    this.classList.add('active');
                    
                    // Animate section transition
                    animateSectionTransition(this.getAttribute('data-section'));
                    
                    // Close sidebar on mobile after navigation
                    if (window.innerWidth <= 768) {
                        document.querySelector('.sidebar').classList.remove('active');
                    }
                });
            });
        }
        
        // Mobile menu toggle
        if (menuToggle) {
            menuToggle.addEventListener('click', function() {
                document.querySelector('.sidebar').classList.toggle('active');
            });
        }
    }
    
    // Animate section transition
    function animateSectionTransition(sectionId) {
        // Fade out all sections
        contentSections.forEach(section => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            section.classList.remove('active');
            
            // Reset any animations inside
            section.querySelectorAll('.reveal-element').forEach(el => {
                el.classList.remove('reveal-animation');
            });
        });
        
        // Update page title
        updatePageTitle(sectionId);
        
        // After a short delay, show the target section with animation
        setTimeout(() => {
            const targetSection = document.getElementById(`${sectionId}-section`);
            targetSection.classList.add('active');
            
            // Staggered animation for elements inside the section
            targetSection.querySelectorAll('.reveal-element').forEach((el, index) => {
                setTimeout(() => {
                    el.classList.add('reveal-animation');
                }, 100 * index);
            });
            
            // Animate the section
            targetSection.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            targetSection.style.opacity = '1';
            targetSection.style.transform = 'translateY(0)';
        }, 300);
    }
    
    // Update page title based on section
    function updatePageTitle(section) {
        if (pageTitle) {
            const newTitle = section.charAt(0).toUpperCase() + section.slice(1);
            
            // Animate the title change
            pageTitle.style.opacity = '0';
            pageTitle.style.transform = 'translateY(-10px)';
            
            setTimeout(() => {
                pageTitle.textContent = newTitle;
                pageTitle.style.opacity = '1';
                pageTitle.style.transform = 'translateY(0)';
            }, 300);
        }
    }

    // Theme toggle functionality
    function initTheme() {
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
            
            // Theme toggle click handler with animation
            themeToggle.addEventListener('click', function() {
                const isDarkMode = !document.body.classList.contains('dark-mode');
                document.body.classList.toggle('dark-mode', isDarkMode);
                updateThemeIcon(isDarkMode);
                localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
            });
        }
    }
    
    // Update theme toggle icon
    function updateThemeIcon(isDarkMode) {
        if (themeToggle) {
            const icon = themeToggle.querySelector('i');
            if (icon) {
                icon.className = isDarkMode ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
    }

    // Initialize camera state
    function initCameraState() {
        // Initialize cameras in inactive state with overlay
        if (drowsinessOverlay && distractionOverlay) {
            showCameraOverlays(true);
            // Update overlay message to indicate monitoring needs to be started
            if (drowsinessOverlay.querySelector('p')) {
                drowsinessOverlay.querySelector('p').textContent = 'Click "Start Monitoring" to begin';
            }
            if (distractionOverlay.querySelector('p')) {
                distractionOverlay.querySelector('p').textContent = 'Click "Start Monitoring" to begin';
            }
        }
        
        // Initialize camera fit mode from localStorage
        const cameraFitMode = document.getElementById('cameraFitMode');
        if (cameraFitMode) {
            const savedFitMode = localStorage.getItem('cameraFitMode') || 'cover';
            cameraFitMode.value = savedFitMode;
            
            // Apply the saved preference to both cameras
            if (drowsinessFeed) drowsinessFeed.className = savedFitMode;
            if (distractionFeed) distractionFeed.className = savedFitMode;
            
            // Save preference when changed
            cameraFitMode.addEventListener('change', function() {
                localStorage.setItem('cameraFitMode', this.value);
                if (drowsinessFeed) drowsinessFeed.className = this.value;
                if (distractionFeed) distractionFeed.className = this.value;
            });
        }

        // Ensure cameras are not active on initial load
        if (drowsinessFeed) drowsinessFeed.src = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';
        if (distractionFeed) distractionFeed.src = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';
    }
    
    function showCameraOverlays(show) {
        // Show or hide camera overlays for both cameras
        if (drowsinessOverlay) {
            drowsinessOverlay.style.display = show ? 'flex' : 'none';
            // Only update message if monitoring is not active
            if (show && !isMonitoringActive) {
                drowsinessOverlay.querySelector('p').textContent = 'Click "Start Monitoring" to begin';
            }
        }
        if (distractionOverlay) {
            distractionOverlay.style.display = show ? 'flex' : 'none';
            // Only update message if monitoring is not active
            if (show && !isMonitoringActive) {
                distractionOverlay.querySelector('p').textContent = 'Click "Start Monitoring" to begin';
            }
        }
    }

    // Activate camera
    function activateCamera() {
        // Call the toggle_monitoring endpoint to ensure backend state is updated
        fetch('/toggle_monitoring', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log("Camera monitoring toggled:", data);
            
            // Hide all overlays immediately
            showCameraOverlays(false);
            
            // Start the camera feeds with a slight delay
            setTimeout(() => {
                if (drowsinessFeed) {
                    const timestamp = new Date().getTime();
                    drowsinessFeed.src = `/drowsiness_feed?t=${timestamp}`;
                    drowsinessFeed.onerror = function() {
                        console.error("Error loading drowsiness camera feed");
                        if (drowsinessOverlay) {
                            drowsinessOverlay.style.display = 'flex';
                            drowsinessOverlay.innerHTML = '<i class="fas fa-exclamation-triangle"></i><p>Camera error</p>';
                        }
                    };
                }
                
                if (distractionFeed) {
                    const timestamp = new Date().getTime();
                    distractionFeed.src = `/distraction_feed?t=${timestamp}`;
                    distractionFeed.onerror = function() {
                        console.error("Error loading distraction camera feed");
                        if (distractionOverlay) {
                            distractionOverlay.style.display = 'flex';
                            distractionOverlay.innerHTML = '<i class="fas fa-exclamation-triangle"></i><p>Camera error</p>';
                        }
                    };
                }
            }, 500);
            
            // Update Camera status indicator
            if (cameraStatus) {
                cameraStatus.querySelector('.status-dot').className = 'status-dot active';
                cameraStatusText.textContent = 'Camera Active';
            }
            
            // Add active effect to status cards
            statusCards.forEach(card => {
                card.classList.add('active');
                addPulseEffect(card);
            });
            
            // Start polling for status updates
            startStatusPolling();
            
            console.log("Camera activated");
        })
        .catch(error => {
            console.error('Error activating camera:', error);
            
            // Show error in overlays
            if (drowsinessOverlay) {
                drowsinessOverlay.style.display = 'flex';
                drowsinessOverlay.innerHTML = '<i class="fas fa-exclamation-triangle"></i><p>Failed to start monitoring</p>';
            }
            
            if (distractionOverlay) {
                distractionOverlay.style.display = 'flex';
                distractionOverlay.innerHTML = '<i class="fas fa-exclamation-triangle"></i><p>Failed to start monitoring</p>';
            }
        });
    }
    
    // Deactivate camera
    function deactivateCamera() {
        // Stop the camera feeds - use empty GIF to prevent network requests
        const emptyGif = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';
        
        if (drowsinessFeed) {
            drowsinessFeed.src = emptyGif;
        }
        
        if (distractionFeed) {
            distractionFeed.src = emptyGif;
        }
        
        // Show camera overlays with appropriate message
        if (drowsinessOverlay && distractionOverlay) {
            showCameraOverlays(true);
            if (drowsinessOverlay.querySelector('p')) {
                drowsinessOverlay.querySelector('p').textContent = 'Click "Start Monitoring" to begin';
            }
            if (distractionOverlay.querySelector('p')) {
                distractionOverlay.querySelector('p').textContent = 'Click "Start Monitoring" to begin';
            }
        }
        
        // Update camera status indicator
        if (cameraStatus) {
            cameraStatus.querySelector('.status-dot').className = 'status-dot inactive';
            cameraStatusText.textContent = 'Camera Inactive';
        }
        
        // Remove active effect from status cards
        statusCards.forEach(card => {
            card.classList.remove('active');
        });
        
        // Stop polling for status updates
        stopStatusPolling();
        
        // Reset status displays
        if (drowsinessLevel) drowsinessLevel.textContent = 'Inactive';
        if (drowsinessDescription) drowsinessDescription.textContent = 'Monitoring paused';
        if (drowsinessProgress) drowsinessProgress.style.width = '0%';
        
        if (distractionLevel) distractionLevel.textContent = 'Inactive';
        if (distractionDescription) distractionDescription.textContent = 'Monitoring paused';
        if (distractionProgress) distractionProgress.style.width = '0%';
        
        console.log("Camera deactivated");
    }

    // Polling interval reference
    let pollingInterval = null;
    
    // Start polling for status updates
    function startStatusPolling() {
        // Initial status update
        updateStatus();
        
        // Clear any existing interval
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }
        
        // Poll every second
        pollingInterval = setInterval(updateStatus, 1000);
    }
    
    // Stop polling for status updates
    function stopStatusPolling() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }

    // Get status from the server
    function updateStatus() {
        fetch('/get_status')
            .then(response => response.json())
            .then(data => {
                updateStatusUI(data);
                checkAndSendLocationIfNeeded(data);
            })
            .catch(error => {
                console.error('Error updating status:', error);
                updateStatusUI({
                    drowsiness: 'Error', 
                    distraction: 'Error', 
                    camera_available: false
                });
            });
    }

    // Update the UI with the latest status
    function updateStatusUI(data) {
        // Skip updating if the data contains error states and we had a valid state before
        if (data.drowsiness === 'Error' || data.distraction === 'Error') {
            if (drowsinessLevel && drowsinessLevel.textContent !== 'Error' && 
                drowsinessLevel.textContent !== 'Inactive' && 
                drowsinessLevel.textContent !== 'Initializing...') {
                return; // Keep the previous valid state
            }
        }
        
        // Update drowsiness status
        if (drowsinessLevel && drowsinessDescription && drowsinessBox) {
            const oldDrowsiness = drowsinessLevel.textContent;
            const newDrowsiness = data.drowsiness;
            
            if (oldDrowsiness !== newDrowsiness) {
                // Only update if we're not going from a valid state to an error state
                if (!(newDrowsiness === 'Error' && oldDrowsiness !== 'Error' && 
                      oldDrowsiness !== 'Inactive' && oldDrowsiness !== 'Initializing...')) {
                    // Animate value change
                    animateStatusChange(drowsinessLevel, newDrowsiness);
                    drowsinessDescription.textContent = getDrowsinessDescription(newDrowsiness);
                    
                    // Update the status card appearance with animation
                    drowsinessBox.classList.remove('status-alert', 'status-warning', 'status-danger');
                    
                    // Update progress bar
                    if (drowsinessProgress) {
                        let progressWidth = '0%';
                        
                        if (newDrowsiness === 'Drowsy') {
                            drowsinessBox.classList.add('status-warning');
                            progressWidth = '65%';
                            addPulseEffect(drowsinessBox);
                        } else if (newDrowsiness === 'Very Drowsy') {
                            drowsinessBox.classList.add('status-danger');
                            progressWidth = '30%';
                            addPulseEffect(drowsinessBox);
                            
                            // Also show an alert for severe drowsiness
                            showDistractionAlert('Severe drowsiness detected! Driver needs immediate attention.');
                        } else if (newDrowsiness !== 'Error') {
                            drowsinessBox.classList.add('status-alert');
                            progressWidth = '95%';
                        }
                        
                        // Animate progress bar
                        drowsinessProgress.style.width = '0%';
                        setTimeout(() => {
                            drowsinessProgress.style.width = progressWidth;
                        }, 300);
                    }
                }
            }
        }
        
        // Update distraction status
        if (distractionLevel && distractionDescription && distractionBox) {
            const oldDistraction = distractionLevel.textContent;
            const newDistraction = data.distraction;
            
            if (oldDistraction !== newDistraction) {
                // Only update if we're not going from a valid state to an error state
                if (!(newDistraction === 'Error' && oldDistraction !== 'Error' && 
                      oldDistraction !== 'Inactive' && oldDistraction !== 'Initializing...')) {
                    // Animate value change
                    animateStatusChange(distractionLevel, newDistraction);
                    distractionDescription.textContent = getDistractionDescription(newDistraction);
                    
                    // Update the status card appearance with animation
                    distractionBox.classList.remove('status-alert', 'status-warning', 'status-danger');
                    
                    // Update progress bar
                    if (distractionProgress) {
                        let progressWidth = '0%';
                        
                        if (["Looking Away", "Adjusting Controls", "Hands Off Steering"].includes(newDistraction)) {
                            distractionBox.classList.add('status-warning');
                            progressWidth = '60%';
                            addPulseEffect(distractionBox);
                            
                            // Show alert for distraction
                            if (newDistraction === "Looking Away") {
                                showDistractionAlert('Driver is looking away from the road!');
                            } else if (newDistraction === "Hands Off Steering") {
                                showDistractionAlert('Driver has taken hands off the steering wheel!');
                            }
                        } else if (["Phone Use", "Eating or Drinking", "Grooming"].includes(newDistraction)) {
                            distractionBox.classList.add('status-danger');
                            progressWidth = '25%';
                            addPulseEffect(distractionBox);
                            
                            // Show alert for dangerous distraction
                            if (newDistraction === "Phone Use") {
                                showDistractionAlert('DANGER! Driver is using a phone while driving!');
                            } else if (newDistraction === "Eating or Drinking") {
                                showDistractionAlert('DANGER! Driver is eating or drinking while driving!');
                            } else if (newDistraction === "Grooming") {
                                showDistractionAlert('DANGER! Driver is grooming while driving!');
                            }
                        } else if (newDistraction !== 'Error') {
                            distractionBox.classList.add('status-alert');
                            progressWidth = '95%';
                            
                            // Hide alert if distraction is resolved
                            hideDistractionAlert();
                        }
                        
                        // Animate progress bar
                        distractionProgress.style.width = '0%';
                        setTimeout(() => {
                            distractionProgress.style.width = progressWidth;
                        }, 300);
                    }
                }
            }
        }
        
        // Update camera status
        if (cameraStatus && cameraStatusText) {
            const statusDot = cameraStatus.querySelector('.status-dot');
            // Always consider camera active during monitoring
            const isActive = isMonitoringActive;
            
            if (statusDot) {
                // Animate status change if needed
                if ((statusDot.classList.contains('active') && !isActive) || 
                    (statusDot.classList.contains('inactive') && isActive)) {
                    
                    statusDot.style.transform = 'scale(1.5)';
                    setTimeout(() => {
                        statusDot.className = isActive ? 'status-dot active' : 'status-dot inactive';
                        statusDot.style.transform = 'scale(1)';
                    }, 300);
                    
                    // Animate text change
                    cameraStatusText.style.opacity = '0';
                    setTimeout(() => {
                        cameraStatusText.textContent = isActive ? 'Camera Active' : 'Camera Inactive';
                        cameraStatusText.style.opacity = '1';
                    }, 300);
                } else {
                    statusDot.className = isActive ? 'status-dot active' : 'status-dot inactive';
                    cameraStatusText.textContent = isActive ? 'Camera Active' : 'Camera Inactive';
                }
            }
            
            // Update camera overlay visibility
            if (drowsinessOverlay && distractionOverlay) {
                // Only show overlays when monitoring is not active
                showCameraOverlays(!isMonitoringActive);
            }
        }
        
        // Show notification for significant changes
        if (isAlertCondition(data.drowsiness, data.distraction)) {
            showStatusNotification(data.drowsiness, data.distraction);
        }
    }
    
    // Animate status value change
    function animateStatusChange(element, newValue) {
        element.style.transform = 'translateY(-10px)';
        element.style.opacity = '0';
        
        setTimeout(() => {
            element.textContent = newValue;
            element.style.transform = 'translateY(0)';
            element.style.opacity = '1';
        }, 300);
    }
    
    // Add pulse effect to a card
    function addPulseEffect(card) {
        card.classList.add('pulse-animation');
        setTimeout(() => {
            card.classList.remove('pulse-animation');
        }, 1000);
    }
    
    // Check if current status should trigger an alert
    function isAlertCondition(drowsiness, distraction) {
        return drowsiness === 'Very Drowsy' || 
               drowsiness === 'Drowsy' || 
               ["Phone Use", "Eating or Drinking", "Grooming", "Looking Away"].includes(distraction);
    }

    // Descriptions for drowsiness states
    function getDrowsinessDescription(val) {
        switch(val) {
            case 'Drowsy': 
                return 'Driver appears to be drowsy. Eyes are closing frequently.';
            case 'Very Drowsy': 
                return 'Driver is showing significant signs of drowsiness. Immediate attention required!';
            case 'Alert': 
                return 'Driver is alert and attentive to the road.';
            case 'Normal': 
                return 'Driver is alert and showing no signs of drowsiness.';
            default: 
                return 'Initializing drowsiness detection system...';
        }
    }
    
    // Descriptions for distraction states
    function getDistractionDescription(val) {
        switch(val) {
            case 'Phone Use': 
                return 'Driver is using a phone while driving. This reduces reaction time significantly.';
            case 'Eating or Drinking': 
                return 'Driver is eating or drinking while driving, which may impact control of the vehicle.';
            case 'Grooming': 
                return 'Driver is engaged in grooming activities, diverting attention from the road.';
            case 'Looking Away': 
                return 'Driver is looking away from the road, reducing awareness of surroundings.';
            case 'Adjusting Controls': 
                return 'Driver is adjusting vehicle controls, temporarily diverting attention.';
            case 'Hands Off Steering': 
                return 'Driver does not have proper control of the steering wheel.';
            case 'Attentive': 
                return 'Driver is attentive to the road with proper focus on driving.';
            case 'Normal': 
                return 'Driver is showing proper attention to the road ahead.';
            default: 
                return 'Initializing distraction detection system...';
        }
    }

    // Show status notification with animation
    function showStatusNotification(drowsiness, distraction) {
        // Determine notification type
        let notificationType = 'notification-info';
        let title = 'Status Update';
        let message = '';
        let icon = 'info-circle';
        
        if (drowsiness === 'Very Drowsy') {
            notificationType = 'notification-danger';
            title = 'Severe Drowsiness Detected';
            message = 'Driver shows signs of extreme drowsiness. Immediate attention required!';
            icon = 'exclamation-triangle';
        } else if (drowsiness === 'Drowsy') {
            notificationType = 'notification-warning';
            title = 'Drowsiness Detected';
            message = 'Driver appears to be drowsy. Please stay alert.';
            icon = 'bell';
        } else if (["Phone Use", "Eating or Drinking", "Grooming"].includes(distraction)) {
            notificationType = 'notification-danger';
            title = 'Dangerous Distraction';
            message = `${distraction} detected. This is unsafe while driving.`;
            icon = 'exclamation-triangle';
        } else if (["Looking Away", "Adjusting Controls", "Hands Off Steering"].includes(distraction)) {
            notificationType = 'notification-warning';
            title = 'Distraction Detected';
            message = `${distraction} detected. Please focus on the road.`;
            icon = 'bell';
        }
        
        // Create notification element with enhanced animation
        const notification = document.createElement('div');
        notification.className = `status-notification ${notificationType}`;
        notification.innerHTML = `
            <div class="notification-icon">
                <i class="fas fa-${icon}"></i>
            </div>
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${message}</div>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add close functionality
        notification.querySelector('.notification-close').addEventListener('click', function() {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }
        }, 5000);
        
        // Add to notification area
        notificationArea.appendChild(notification);
    }

    // Add ripple effect to buttons
    function addRippleEffect(button) {
        // Remove any existing ripple
        const ripple = button.querySelector('.ripple');
        if (ripple) {
            ripple.remove();
        }
        
        // Create ripple element
        const rippleEl = document.createElement('span');
        rippleEl.className = 'ripple';
        button.appendChild(rippleEl);
        
        // Position and animate
        const buttonRect = button.getBoundingClientRect();
        const diameter = Math.max(buttonRect.width, buttonRect.height);
        const radius = diameter / 2;
        
        rippleEl.style.width = rippleEl.style.height = `${diameter}px`;
        rippleEl.style.left = '0px';
        rippleEl.style.top = '0px';
        rippleEl.style.transform = 'scale(0)';
        
        // Trigger animation
        setTimeout(() => {
            rippleEl.style.transform = 'scale(2)';
            rippleEl.style.opacity = '0';
            
            // Remove after animation completes
            setTimeout(() => {
                rippleEl.remove();
            }, 600);
        }, 10);
    }

    // Camera Controls and Zoom functionality
    function setupCameraControls() {
        // Get all camera control buttons for both feeds
        const drowsinessZoomIn = document.getElementById('drowsinessZoomIn');
        const drowsinessZoomOut = document.getElementById('drowsinessZoomOut');
        const drowsinessResetZoom = document.getElementById('drowsinessResetZoom');
        
        const distractionZoomIn = document.getElementById('distractionZoomIn');
        const distractionZoomOut = document.getElementById('distractionZoomOut');
        const distractionResetZoom = document.getElementById('distractionResetZoom');
        
        // Variables to track zoom level for each camera
        let drowsinessZoomLevel = 1;
        let distractionZoomLevel = 1;
        
        // Setup drowsiness camera controls
        if (drowsinessZoomIn && drowsinessZoomOut && drowsinessResetZoom && drowsinessFeed) {
            drowsinessZoomIn.addEventListener('click', function() {
                drowsinessZoomLevel = Math.min(drowsinessZoomLevel + 0.1, 2);
                updateDrowsinessZoom();
            });
            
            drowsinessZoomOut.addEventListener('click', function() {
                drowsinessZoomLevel = Math.max(drowsinessZoomLevel - 0.1, 1);
                updateDrowsinessZoom();
            });
            
            drowsinessResetZoom.addEventListener('click', function() {
                drowsinessZoomLevel = 1;
                updateDrowsinessZoom();
            });
            
            function updateDrowsinessZoom() {
                drowsinessFeed.style.transform = `scale(${drowsinessZoomLevel})`;
            }
        }
        
        // Setup distraction camera controls
        if (distractionZoomIn && distractionZoomOut && distractionResetZoom && distractionFeed) {
            distractionZoomIn.addEventListener('click', function() {
                distractionZoomLevel = Math.min(distractionZoomLevel + 0.1, 2);
                updateDistractionZoom();
            });
            
            distractionZoomOut.addEventListener('click', function() {
                distractionZoomLevel = Math.max(distractionZoomLevel - 0.1, 1);
                updateDistractionZoom();
            });
            
            distractionResetZoom.addEventListener('click', function() {
                distractionZoomLevel = 1;
                updateDistractionZoom();
            });
            
            function updateDistractionZoom() {
                distractionFeed.style.transform = `scale(${distractionZoomLevel})`;
            }
        }
    }

    // Show distraction alert banner
    function showDistractionAlert(message) {
        if (!isMonitoringActive) return; // Don't show alerts if monitoring is not active
        
        if (distractionAlert && alertMessage) {
            distractionAlert.style.display = 'flex';
            alertMessage.textContent = message;
            distractionAlert.classList.add('show');
            
            // Add delay before playing alert sound
            setTimeout(() => {
                playAlertSound();
            }, 1000); // 1 second delay
            
            addPopAnimation();
        }
    }
    
    // Hide distraction alert banner
    function hideDistractionAlert() {
        if (distractionAlert) {
            // Fade out animation
            distractionAlert.style.opacity = '0';
            distractionAlert.style.transform = 'scale(0.9)';
            
            // Hide after animation completes
            setTimeout(() => {
                distractionAlert.style.display = 'none';
            }, 300);
        }
    }
    
    // Play alert sound
    function playAlertSound() {
        if (!isMonitoringActive) return; // Don't play sound if monitoring is not active
        
        const audio = new Audio('/static/sounds/alert.mp3');
        audio.volume = 0.3; // Reduced volume
        audio.playbackRate = 0.8; // Slower playback rate
        audio.play().catch(error => {
            console.log('Audio playback failed:', error);
        });
    }
    
    // Add pop animation to CSS
    function addPopAnimation() {
        const styleSheet = document.createElement('style');
        styleSheet.textContent = `
            @keyframes alert-pop {
                0% {
                    opacity: 0;
                    transform: scale(0.9);
                }
                70% {
                    opacity: 1;
                    transform: scale(1.05);
                }
                100% {
                    opacity: 1;
                    transform: scale(1);
                }
            }
        `;
        document.head.appendChild(styleSheet);
    }
    
    // Call this on initialization
    addPopAnimation();

    // Add polling for pending_gps_email (to be added in /get_status response)
    function checkAndSendLocationIfNeeded(data) {
        if (data.pending_gps_email) {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(pos) {
                    fetch('/api/send-location', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            latitude: pos.coords.latitude,
                            longitude: pos.coords.longitude
                        })
                    })
                    .then(res => res.json())
                    .then(resp => {
                        if (resp.success) {
                            showStatusNotification(`Emergency email sent to ${resp.email}`);
                        }
                    });
                }, function(err) {
                    showStatusNotification('Unable to get GPS location for emergency email.');
                });
            } else {
                showStatusNotification('Geolocation not supported.');
            }
        }
    }
}); 