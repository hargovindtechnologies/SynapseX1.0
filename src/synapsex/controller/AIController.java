package com.synapsex.controller;

import com.synapsex.service.AIService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class AIController {

    private static final Set<String> VALID_KEYS = Set.of("YOUR_API_KEY_1","YOUR_API_KEY_2");

    @Autowired
    private AIService aiService;

    /**
     * Example POST request:
     * POST /api/predict
     * body: {"input": [[0.5,0.2,0.1,0.3],[0.1,0.2,0.3,0.4]]}
     */
    @PostMapping("/predict")
    public PredictionResponse predict(@RequestHeader("x-api-key") String key,
                                      @RequestBody PredictionRequest request) {
        if (!VALID_KEYS.contains(key)) throw new RuntimeException("Invalid API key");
        double[][] output = aiService.predict(request.input);
        return new PredictionResponse(output);
    }

    // DTOs
    public static class PredictionRequest {
        public double[][] input;
    }

    public static class PredictionResponse {
        public double[][] output;
        public PredictionResponse(double[][] output) { this.output = output; }
    }
}
