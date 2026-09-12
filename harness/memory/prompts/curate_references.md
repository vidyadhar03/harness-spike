## Task: group reference images into visual directions
You are given the memory context for one location, then a numbered set of candidate images retrieved from an archive. Each image comes with its index, title, and description, followed by the image itself.

The director is brainstorming and has not chosen anything yet, so do not rank the images into one best answer. Group them into distinct visual directions the production could take.

- directions: 2 to 5 groups.
  - name: 3 to 6 words naming the direction in visual terms, not archive terms ("Weathered slate and dark timber", "Open terraced slopes").
  - why: one or two sentences on what characterises this direction and how it differs from the others. Say plainly where it departs from the context, because a departure the director can see is useful.
  - images: the indices in this direction, most representative first. An image belongs to exactly one direction.
- captions: one entry per image you keep.
  - index: the image's index.
  - caption: 1 to 3 sentences describing what is actually visible: place type, layout, materials, condition, light, time of day, and scale cues. Describe the image, not the script. Never state that this image shows the production's location; it is a reference for how the place could look.
- Drop images that are maps, diagrams, logos, portraits, close-ups of objects with no sense of place, or that show nothing relevant to the context. Leaving an image out of every direction drops it.
- A direction needs at least 2 images. If fewer than 2 images are usable overall, return no directions.
