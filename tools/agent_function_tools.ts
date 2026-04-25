import { FunctionTool } from '@google/adk';
import { z } from 'zod';

/**
 * A mock tool that returns a hardcoded time for a given city.
 */
export const getCurrentTime = new FunctionTool({
  name: 'get_current_time',
  description: 'Returns the current time in a specified city.',
  parameters: z.object({
    city: z.string().describe('The name of the city for which to retrieve the current time.'),
  }),
  execute: ({ city }) => {
    return { status: 'success', report: `The current time in ${city} is 10:30 AM` };
  },
});
