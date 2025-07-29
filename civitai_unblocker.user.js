// ==UserScript==
// @name         Civitai Blocked Image Revealer (with nth-child)
// @namespace    http://tampermonkey.net/
// @version      1.1
// @description  Monitors for "Blocked Image" placeholders on Civitai and replaces them with the actual generated image by fetching the job result.
// @author       You
// @match        https://civitai.com/*
// @connect      orchestration.civitai.com
// @grant        GM_xmlhttpRequest
// ==/UserScript==



(function() {
    'use strict';
    const CIVITAI_API_TOKEN = 'API_KEY_HERE'

    // =========================================================================
    // HOW TO CONFIGURE: Find the Job ID
    // =========================================================================
    // This script needs to know the 'jobId' to fetch the image.
    // By default, it tries to find it in the URL (e.g., civitai.com/jobs/YOUR_JOB_ID).
    // If the jobId is located elsewhere, you MUST update the `getJobId` function below.
    //
    function getJobIds(mainDiv) {
        return new URL(mainDiv.closest('div[data-with-border="true"]').querySelector('a[href*="clickup"]').href).searchParams.get('Job IDs').split(',');
    }


    /**
     * This is the JavaScript function from the previous request, converted to use
     * Tampermonkey's GM_xmlhttpRequest for cross-domain requests.
     * @param {string} jobId - The ID of the job to retrieve.
     * @returns {Promise<string[]>} A promise that resolves to an array of blob URLs.
     */
    function getV1ConsumerJob(jobId, detailed = null) {
        return new Promise((resolve, reject) => {
            if (!jobId) {
                return reject("jobId cannot be null.");
            }


            const url = `https://orchestration.civitai.com/v1/consumer/jobs/${jobId}`;
            console.log('fetching url', url);

            GM_xmlhttpRequest({
                method: "GET",
                url: url,
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    // NOTE: This bearer token is from your example.
                    "Authorization": "Bearer " + CIVITAI_API_TOKEN,
                },
                onload: function(response) {
                    if (response.status >= 200 && response.status < 300) {
                        const data = JSON.parse(response.responseText);
                        if (data && data.result && Array.isArray(data.result)) {

                            const imageUrls = data.result
                                .filter(item => item && item.available && item.blobUrl)
                                .map(item => item.blobUrl);
                            resolve(imageUrls);
                        } else {
                            resolve([]); // Resolve with empty if structure is unexpected
                        }
                    } else {
                        reject(`API request failed with status ${response.status}: ${response.responseText}`);
                    }
                },
                onerror: function(error) {
                    reject(error);
                }
            });
        });
    }


    /**
     * This function performs the DOM manipulation to replace the placeholder.
     * @param {HTMLElement} pElement - The <p> element with "Blocked Image" text.
     */
    async function replaceBlockedImage(pElement) {
        // Mark the element as being processed to avoid running the function twice on it.
        if (pElement.dataset.processed) return;

        const blockedDiv = pElement.parentElement;
        blockedDiv.innerText = 'unblocking image...';
        const mainDiv = blockedDiv.parentElement;

        if (!mainDiv) {
            console.warn("Could not find a parent element to attach the image to.");
            return;
        }

        // ======================= YOUR CHANGE IS HERE ==========================
        // Find the order of mainDiv within its parent element.
        const childIdx = Array.from(mainDiv.parentElement.children).indexOf(mainDiv);
        // ======================================================================

        try {
            const jobIds = getJobIds(mainDiv);
            const imgUrls = (await Promise.all(jobIds.map(getV1ConsumerJob))).flat();

            if (!imgUrls || imgUrls.length === 0) {
                pElement.innerText = "Image not available.";
                pElement.style.color = "orange";
                console.log("No available image URLs were returned from the API for jobs:", jobIds);
                return;
            }

            // Check if childIdx is valid and within the bounds of available images
            if (childIdx < 0 || childIdx >= imgUrls.length) {
                console.warn(`childIdx ${childIdx} is out of bounds for imgUrls array of length ${imgUrls.length}`);
                pElement.innerText = "Image index out of range.";
                pElement.style.color = "orange";
                return;
            }

            // Get the specific image URL for this child index
            const targetUrl = imgUrls[childIdx];
            if (!targetUrl) {
                console.warn(`No image URL found at index ${childIdx}`);
                pElement.innerText = "Image not found at expected position.";
                pElement.style.color = "orange";
                return;
            }

            // Create the image element for the target URL
            const link = document.createElement('a');
            link.href = targetUrl;
            link.target = "_blank"; // Open in a new tab when clicked
            link.rel = "noopener noreferrer";
            link.className = "EdgeImage_image__iH4_q max-h-full min-h-0 w-auto max-w-full cursor-pointer";

            const image = document.createElement('img');
            image.src = targetUrl;
            image.className = "EdgeImage_image__iH4_q max-h-full min-h-0 w-auto max-w-full";
            image.alt = `Job Result ${jobIds[0]}`;
            image.style.display = 'block';

            link.appendChild(image);
            mainDiv.insertBefore(link, blockedDiv);

            // Remove the original "Blocked Image" placeholder
            blockedDiv.remove();

        } catch (error) {
            console.error("Failed to replace blocked image:", error);
            pElement.innerText = "Error loading image.";
            pElement.style.color = "red";
        }
        pElement.dataset.processed = "true";

    }


    /**
     * The main function that sets up the observer and runs the initial check.
     */
    function main() {
        // Create an observer to watch for DOM changes
        const observer = new MutationObserver((mutationsList) => {
            for (const mutation of mutationsList) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach(node => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            if (node.matches('p.mantine-focus-auto') && node.innerText.toLowerCase().includes("image")) {
                                console.log('replacing', p);
                                replaceBlockedImage(node);
                            }
                            const targets = node.querySelectorAll('p.mantine-focus-auto');
                            targets.forEach(p => {
                                if (p.innerText.toLowerCase().includes("image")) {
                                    console.log('replacing', p);
                                    replaceBlockedImage(p);
                                }
                            });
                        }
                    });
                }
            }
        });

        // Start observing the entire document body for additions of new elements
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Also, run an initial check in case the elements are already on the page
        const initialElements = document.querySelectorAll('p.mantine-focus-auto');
        initialElements.forEach(p => {
            if (p.innerText.toLowerCase().includes("image")) {
                console.log('replacing', p);
                replaceBlockedImage(p);
            }
        });

        console.log("Tampermonkey Script: Blocked Image Revealer is active and watching for changes.");
    }

    // Run the main function
    main();

})();
