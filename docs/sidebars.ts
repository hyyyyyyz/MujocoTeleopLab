import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/offline-sim2sim',
        'tutorials/pico-sim2sim',
        'tutorials/pico-sim2real',
        'tutorials/high-level-policy-sim2real',
        'tutorials/training',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        {
          type: 'category',
          label: 'Configuration',
          items: [
            'reference/configuration/overview',
            'reference/configuration/fields',
          ],
        },
        'reference/architecture',
        {
          type: 'category',
          label: 'Resources',
          items: [
            'reference/resources/assets',
            'reference/resources/motion-datasets',
            'reference/resources/teleoperation-datasets',
          ],
        },
        'reference/companion-projects',
        'contributing',
      ],
    },
  ],
};

export default sidebars;
