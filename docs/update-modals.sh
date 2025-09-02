#!/bin/bash

# This script replaces the modals section in index.html with updated content

# First, backup the original file
cp index.html index.html.backup

# Find the line where modals start and end
MODAL_START=$(grep -n "<!-- Modals -->" index.html | cut -d: -f1)
SCRIPT_END=$(grep -n "</script>" index.html | tail -1 | cut -d: -f1)

# Create a new file with everything before modals
head -n $((MODAL_START - 1)) index.html > index-new.html

# Add the updated modals content
cat modals-update.html >> index-new.html

# Add the existing JavaScript functions (we need to preserve these)
echo "" >> index-new.html
echo "    <script>" >> index-new.html
echo "        function openModal(id) {" >> index-new.html
echo "            document.getElementById(id + '-modal').style.display = 'block';" >> index-new.html
echo "            document.body.style.overflow = 'hidden';" >> index-new.html
echo "        }" >> index-new.html
echo "" >> index-new.html
echo "        function closeModal(id) {" >> index-new.html
echo "            document.getElementById(id + '-modal').style.display = 'none';" >> index-new.html
echo "            document.body.style.overflow = 'auto';" >> index-new.html
echo "        }" >> index-new.html
echo "" >> index-new.html
echo "        // Close modal when clicking outside" >> index-new.html
echo "        document.addEventListener('click', function(event) {" >> index-new.html
echo "            const modals = document.querySelectorAll('.modal');" >> index-new.html
echo "            modals.forEach(modal => {" >> index-new.html
echo "                if (event.target === modal) {" >> index-new.html
echo "                    modal.style.display = 'none';" >> index-new.html
echo "                    document.body.style.overflow = 'auto';" >> index-new.html
echo "                }" >> index-new.html
echo "            });" >> index-new.html
echo "        });" >> index-new.html
echo "    </script>" >> index-new.html
echo "</body>" >> index-new.html
echo "</html>" >> index-new.html

# Replace the original file
mv index-new.html index.html

echo "Modals section has been updated successfully!"