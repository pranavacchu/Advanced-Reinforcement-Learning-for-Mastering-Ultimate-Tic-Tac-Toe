document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('rulesModal');
    const openModalBtn = document.querySelector('.learn-rules-btn');
    const closeModalBtn = document.querySelector('.close-modal');

    // Open modal
    openModalBtn.addEventListener('click', () => {
        modal.style.display = 'flex';
        // Use setTimeout to ensure the display: flex is applied before adding the active class
        setTimeout(() => {
            modal.classList.add('active');
        }, 10);
    });

    // Close modal functions
    const closeModal = () => {
        modal.classList.remove('active');
        setTimeout(() => {
            modal.style.display = 'none';
        }, 300); // Match the transition duration
    };

    // Close with button
    closeModalBtn.addEventListener('click', closeModal);

    // Close with escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeModal();
        }
    });

    // Close when clicking outside the modal
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Prevent closing when clicking inside modal content
    modal.querySelector('.modal-content').addEventListener('click', (e) => {
        e.stopPropagation();
    });
}); 